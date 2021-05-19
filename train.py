import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
import random
import copy

import configs
from Logger import Logger
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file


def test(val_loader, model, params):
    model.eval()
    acc = model.test_loop(params.save_iter, val_loader, params)
    return acc

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):
    if not params.ad_align:
        if optimization == 'Adam':
            optimizer = torch.optim.Adam(model.parameters())
        else:
            raise ValueError('Unknown optimization, please define by yourself')
    else:
        param_ids = [id(param) for param in model.discriminator]
        base_params = [param for param in model.parameters() if id(param) not in param_ids]
        base_optimizer = torch.optim.Adam(base_params)
        discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters())
        optimizer = [base_optimizer, discriminator_optimizer]

    max_acc = 0
    ori_base_loader = base_loader
    for epoch in range(start_epoch,stop_epoch):
        if params.pseudo_align:
            model.train()
            base_loader = model.get_pseudo_samples(ori_base_loader[:2], params)
            if params.simclr or (params.pseudo_align and params.classcontrast):
                base_loader = list(base_loader)
                base_loader.append(ori_base_loader[-1])

        model.train()
        end = time.time()
        loss = model.train_loop(epoch, base_loader,  optimizer, params) #model are called by reference, no need to return
        print(f'Training time: {time.time() - end:.0f} s')
        if not isinstance(loss, dict):
            params.logger.scalar_summary('train/loss', loss, epoch)
        else:
            for key, value in loss.items():
                params.logger.scalar_summary(f'train/{key}', value, epoch)

        model.eval()
        acc = model.test_loop(epoch, val_loader, params)
        params.logger.scalar_summary('test/acc', acc, epoch)
        if acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

    return model


if __name__ == '__main__':
    params = parse_args('train')
    # DEBUG
    # params.exp = 'debug'
    # params.gpu = '7'
    # params.method = 'baseline'
    # params.loss_type = 'euclidean'
    # params.ad_align = True
    # params.pseudo_align = True
    # params.soft_label = True
    # params.threshold = 0
    # params.init_teacher = 'checkpoints/DomainNet/painting/ResNet18_baseline/0/80.tar'
    # params.momentum = 1
    # params.target_trans = 'randaug'
    # params.simclr = True
    # params.fixmatch = True
    # params.threshold = 0.5
    # params.distribution_align = True
    # params.distribution_m = 1
    # params.fixmatch_gt = True
    # params.fixmatch_anchor = 2
    # params.classcontrast = True
    # params.classcontrast_fn = 'kl'
    # params.pseudomix = True
    # params.pseudomix_fn = 'cutmix'
    # params.rot_align = False
    # params.proto_align = True
    # params.weight_proto = True
    # params.ada_proto = False
    # params.batch_size = 128
    # params.resume = True
    # params.checkpoint = 'checkpoints/DomainNet/painting/ResNet18_baseline/pseudomix_th0.9_a0.7_classcontrast_th0.5_fndot_t0.3'
    # params.checkpoint = 'checkpoints/DomainNet/painting/ResNet18_baseline/0'
    # params.save_iter = 30
    # params.test = True
    # params.cross_domain = 'real'
    # params.split = 'base'
    # params.n_episode = 100

    os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
    if params.seed >= 0:
        np.random.seed(params.seed)
        random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.backends.cudnn.deterministic = True

    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json'
        val_file = configs.data_dir['CUB'] + 'val.json'
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json'
        val_file = configs.data_dir['emnist'] + 'val.json'
    elif params.dataset == 'DomainNet':
        base_folder = '../dataset/DomainNet/real/base/'

        if params.supervised_align:
            base_folder = [base_folder,
                           f'../dataset/DomainNet/{params.cross_domain}/base/']
        val_folder = os.path.join('../dataset/DomainNet/real/',
                                   params.split)
        if params.cross_domain:
            val_folder = [os.path.join(f'../dataset/DomainNet/{params.cross_domain}/',
                                       params.split),
                          # os.path.join(f'../dataset/DomainNet/{params.cross_domain}/',
                          #              params.split),
                          val_folder,
                          # val_folder
                          ]
            if params.reverse_sq:
                val_folder = val_folder[::-1]
        if params.cross_domain and (
                params.ad_align or params.pseudo_align or params.pseudomix or params.proto_align or params.fixmatch or params.classcontrast):
            unlabeled_folder = [f'../dataset/DomainNet/{params.cross_domain}/base',
                                f'../dataset/DomainNet/{params.cross_domain}/val',
                                f'../dataset/DomainNet/{params.cross_domain}/novel']

    else:
        base_file = configs.data_dir[params.dataset] + 'base.json'
        val_file = configs.data_dir[params.dataset] + 'val.json'

    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug, 'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'

    optimization = 'Adam'

    if params.stop_epoch == -1:
        if params.method in ['baseline', 'baseline++']:
            if params.dataset in ['omniglot', 'cross_char']:
                params.stop_epoch = 5
            elif params.dataset in ['CUB']:
                params.stop_epoch = 200  # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
            elif params.dataset in ['miniImagenet', 'cross']:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 400  # default
        else:  # meta-learning methods
            if params.n_shot == 1:
                params.stop_epoch = 600
            elif params.n_shot == 5:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 600  # default

    if params.method in ['baseline', 'baseline++']:
        base_datamgr = SimpleDataManager(image_size, batch_size=params.batch_size)
        base_loader = base_datamgr.get_data_loader(data_folder=base_folder, aug=params.train_aug, rot=params.rot_align, drop_last=True)
        few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(image_size, n_query=15, n_episode=params.n_episode, **few_shot_params)
        val_loader = val_datamgr.get_data_loader(data_folder=val_folder, aug=False, fix_seed=True)
        if params.cross_domain and (
                params.ad_align or params.pseudo_align or params.proto_align or params.fixmatch or params.classcontrast
                or params.pseudomix):
            if params.fixmatch:
                unlabeled_datamgr = SimpleDataManager(image_size, batch_size=params.fixmatch_bs)
                unlabeled_loader = unlabeled_datamgr.get_data_loader(data_folder=unlabeled_folder, fixmatch_trans=True,
                                                                     fixmatch_anchor=params.fixmatch_anchor,
                                                                     add_label=True,
                                                                     proportion=params.unlabeled_proportion, )
            elif params.classcontrast:
                unlabeled_datamgr = SimpleDataManager(image_size, batch_size=params.batch_size)
                unlabeled_loader = unlabeled_datamgr.get_data_loader(data_folder=unlabeled_folder,
                                                                     fixmatch_trans=True,
                                                                     augtype=params.classcontrast_augtype,
                                                                     # fixmatch_anchor=2, fixmatch_weak=False, simclr_trans=True,
                                                                     add_label=True,
                                                                     proportion=params.unlabeled_proportion, )
            else:
                unlabeled_datamgr = SimpleDataManager(image_size, batch_size=params.batch_size)
                unlabeled_loader = unlabeled_datamgr.get_data_loader(data_folder=unlabeled_folder, aug=params.train_aug,
                                                                     add_label=True,
                                                                     proportion=params.unlabeled_proportion,
                                                                     with_idx=True,
                                                                     rot=params.rot_align)
            base_loader = [base_loader, unlabeled_loader]

            if params.simclr:
                simclr_datamgr = SimpleDataManager(image_size, batch_size=params.simclr_bs)
                simclr_loader = simclr_datamgr.get_data_loader(data_folder=unlabeled_folder, simclr_trans=True,
                                                               proportion=params.unlabeled_proportion)
                base_loader.append(simclr_loader)
            if (params.pseudo_align or params.pseudomix) and params.classcontrast:
                unlabeled_datamgr = SimpleDataManager(image_size, batch_size=params.batch_size)
                unlabeled_loader = unlabeled_datamgr.get_data_loader(data_folder=unlabeled_folder, aug=params.train_aug,
                                                                     add_label=True,
                                                                     proportion=params.unlabeled_proportion,
                                                                     with_idx=True,
                                                                     rot=params.rot_align)
                base_loader.insert(1, unlabeled_loader)

        if params.dataset == 'omniglot':
            assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
        if params.dataset == 'cross_char':
            assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'

        if params.method == 'baseline':
            model = BaselineTrain(params, model_dict[params.model], params.num_classes,
                                  ad_align=params.ad_align, ad_loss_weight=params.ad_loss_weight,
                                  pseudo_align=params.pseudo_align, momentum=params.momentum,
                                  threshold=params.threshold, proto_align=params.proto_align,
                                  ada_proto=params.ada_proto, rot_align=params.rot_align)
        elif params.method == 'baseline++':
            model = BaselineTrain(params, model_dict[params.model], params.num_classes, loss_type=params.loss_type,
                                  ad_align=params.ad_align, ad_loss_weight=params.ad_loss_weight,
                                  pseudo_align=params.pseudo_align, momentum=params.momentum,
                                  threshold=params.threshold, proto_align=params.proto_align,
                                  ada_proto=params.ada_proto, rot_align=params.rot_align,
                                  scale=params.scale)

    elif params.method in ['protonet', 'matchingnet', 'relationnet', 'relationnet_softmax', 'maml', 'maml_approx']:
        n_query = max(1,
                      int(15 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
        base_datamgr = SetDataManager(image_size, n_query=n_query, **train_few_shot_params)
        if params.supervised_align:
            base_folder = [base_folder[1], base_folder[0]]
        base_loader = base_datamgr.get_data_loader(data_folder=base_folder, aug=params.train_aug, fix_seed=False)

        test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(image_size, n_query=15, n_episode=params.n_episode, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(data_folder=val_folder, aug=False, fix_seed=True)
        # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

        if params.method == 'protonet':
            model = ProtoNet(model_dict[params.model], **train_few_shot_params)
        elif params.method == 'matchingnet':
            model = MatchingNet(model_dict[params.model], **train_few_shot_params)
        elif params.method in ['relationnet', 'relationnet_softmax']:
            if params.model == 'Conv4':
                feature_model = backbone.Conv4NP
            elif params.model == 'Conv6':
                feature_model = backbone.Conv6NP
            elif params.model == 'Conv4S':
                feature_model = backbone.Conv4SNP
            else:
                feature_model = lambda: model_dict[params.model](flatten=False)
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

            model = RelationNet(feature_model, loss_type=loss_type, **train_few_shot_params)
        elif params.method in ['maml', 'maml_approx']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            model = MAML(model_dict[params.model], approx=(params.method == 'maml_approx'), **train_few_shot_params)
            if params.dataset in ['omniglot', 'cross_char']:  # maml use different parameter in omniglot
                model.n_task = 32
                model.task_update_num = 1
                model.train_lr = 0.1
    else:
        raise ValueError('Unknown method')

    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s/%s_%s' % (
    configs.save_dir, params.dataset, params.cross_domain, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method in ['baseline', 'baseline++']:
        params.checkpoint_dir += '_%dway_%dshot' % (params.train_n_way, params.n_shot)
    params.checkpoint_dir += f'/{params.exp}'

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    log_dir = params.checkpoint_dir.replace('checkpoints', 'tensorboard')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    params.logger = Logger(log_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method == 'maml' or params.method == 'maml_approx':
        stop_epoch = params.stop_epoch * model.n_task  # maml use multiple tasks in one update

    if params.init_model:
        tmp = torch.load(params.init_model)
        model.load_state_dict(tmp['state'], strict=False)

    if (params.pseudo_align or params.pseudomix) and params.init_teacher:
        tmp = torch.load(params.init_teacher)
        feature = copy.deepcopy(model.feature)
        classifier = copy.deepcopy(model.classifier)
        print(f'init teacher from {params.init_teacher}')
        model.load_state_dict(tmp['state'], strict=False)
        model.init_teacher()
        if params.init_student == 'none':
            model.feature = feature
            model.classifier = classifier
        elif params.init_student == 'feature':
            print(f'init student feature from {params.init_teacher}')
            model.classifier = classifier
        elif params.init_student == 'all':
            print(f'init student feature and classifier from {params.init_teacher}')

    # DEBUG
    # source_acc_fc = []
    # source_acc_proto = []
    # target_acc_fc = []
    # target_acc_proto = []
    # target_diff = []
    # for save_iter in range(0, 100, 10):
    #     checkpoint_dir = params.checkpoint
    #     resume_file = get_resume_file(checkpoint_dir, save_iter)
    #     tmp = torch.load(resume_file)
    #     model.load_state_dict(tmp['state'], strict=False)
    #     acc_fc = 0.
    #     proto = [0] * model.num_class
    #     sample_num = [0] * model.num_class
    #     with torch.no_grad():
    #         for x, y in base_loader[0]:
    #             x, y = x.cuda(), y.cuda()
    #             fx = model.feature(x)
    #             score = model.classifier(fx)
    #             pred = score.max(dim=-1)[1]
    #             acc_fc += float((pred==y).sum().item()) / pred.shape[0]
    #             for i in range(fx.shape[0]):
    #                 proto[y[i]] += fx[i]
    #                 sample_num[y[i]] += 1
    #     acc_fc /= len(base_loader[0])
    #     for i in range(model.num_class):
    #         proto[i] /= sample_num[i]
    #     proto = torch.stack(proto)
    #     acc_proto = 0.
    #     with torch.no_grad():
    #         for x, y in base_loader[0]:
    #             x, y = x.cuda(), y.cuda()
    #             fx = model.feature(x)
    #             score = - ((fx.unsqueeze(1) - proto.unsqueeze(0)) ** 2).sum(dim=-1)
    #             pred = score.max(dim=-1)[1]
    #             acc_proto += float((pred == y).sum().item()) / pred.shape[0]
    #     acc_proto /= len(base_loader[0])
    #     source_acc_fc.append(acc_fc)
    #     source_acc_proto.append(acc_proto)
    #
    #     acc_fc, acc_proto, diff = 0., 0., 0.
    #     with torch.no_grad():
    #         for x, y, *_ in unlabeled_loader:
    #             x, y = x[0].cuda(), y.cuda()
    #             fx = model.feature(x)
    #             score = model.classifier(fx)
    #             pred = score.max(dim=-1)[1]
    #             acc_fc += float((pred == y).sum().item()) / pred.shape[0]
    #             score = - ((fx.unsqueeze(1) - proto.unsqueeze(0)) ** 2).sum(dim=-1)
    #             pred2 = score.max(dim=-1)[1]
    #             acc_proto += float((pred2 == y).sum().item()) / pred2.shape[0]
    #             diff += float((pred2 == pred).sum().item()) / pred.shape[0]
    #     acc_fc /= len(unlabeled_loader)
    #     acc_proto /= len(unlabeled_loader)
    #     diff /= len(unlabeled_loader)
    #     target_acc_fc.append(acc_fc)
    #     target_acc_proto.append(acc_proto)
    #     target_diff.append(diff)
    #     print(f'source_acc_fc: {source_acc_fc}')
    #     print(f'source_acc_proto: {source_acc_proto}')
    #     print(f'target_acc_fc: {target_acc_fc}')
    #     print(f'target_acc_proto: {target_acc_proto}')
    #     print(f'target_diff: {target_diff}')
    # exit()

    #     distribution = 0.
    #     with torch.no_grad():
    #         for ux, _ in unlabeled_loader:
    #             ux = ux[0].cuda()
    #             pred = model.classifier(model.feature(ux))
    #             pred = torch.nn.functional.softmax(pred, dim=-1)
    #             distribution += pred.mean(0)
    #         distribution /= len(unlabeled_loader)
    #         entropy = - (distribution * torch.log(distribution)).sum()
    #         entropy_seq.append(entropy.item())
    #         print(entropy_seq)
    # exit()

    # target_acc = []
    # save_iters = [-1] + [x for x in range(0, 100, 10)]
    # for save_iter in save_iters:
    #     checkpoint_dir = params.checkpoint
    #     resume_file = get_resume_file(checkpoint_dir, save_iter)
    #     tmp = torch.load(resume_file)
    #     model.load_state_dict(tmp['state'], strict=False)
    #     acc_fc = 0.
    #     with torch.no_grad():
    #         for x, y, *_ in unlabeled_loader:
    #             x, y = x.cuda(), y.cuda()
    #             fx = model.feature(x)
    #             score = model.classifier(fx)
    #             pred = score.max(dim=-1)[1]
    #             acc_fc += float((pred == y).sum().item()) / pred.shape[0]
    #     acc_fc /= len(unlabeled_loader)
    #     target_acc.append(acc_fc)
    #     print(f'target_acc: {target_acc}')
    # exit()

    if params.resume:
        if params.checkpoint:
            checkpoint_dir = params.checkpoint
        else:
            checkpoint_dir = params.checkpoint_dir
        resume_file = get_resume_file(checkpoint_dir, save_iter=params.save_iter)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            print(f'load state dict from epoch {tmp["epoch"]}')
            start_epoch = tmp['epoch'] + 1
            model.load_state_dict(tmp['state'], strict=False)
            if params.proto_align:
                model.init_teacher()
    elif params.warmup:  # We also support warmup from pretrained baseline feature, but we never used in our paper
        baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (
        configs.save_dir, params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None:
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.",
                                         "")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    if not params.test:
        model = train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params)
    else:
        test(val_loader, model, params)
