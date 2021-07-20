import pickle

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
    if optimization == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
        scheduler = None
    elif optimization == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, params.stop_epoch, eta_min=1e-6)
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0
    for epoch in range(start_epoch,stop_epoch):
        model.train()
        end = time.time()
        loss = model.train_loop(epoch, base_loader, optimizer, params) #model are called by reference, no need to return
        if scheduler is not None:
            scheduler.step()
        print(f'Training time: {time.time() - end:.0f} s')
        if not isinstance(loss, dict):
            params.logger.scalar_summary('train/loss', loss, epoch)
        else:
            for key, value in loss.items():
                params.logger.scalar_summary(f'train/{key}', value, epoch)

        if epoch % params.test_freq == 0:
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
    # params.gpu = '0'
    # params.cross_domain = 'clipart'
    # params.method = 'baseline'
    # params.loss_type = 'euclidean'
    # params.pseudo_align = True
    # params.soft_label = True
    # params.threshold = 0.5
    # params.startup = True
    # params.bn_align = True
    # params.bn_align_mode = 'dsconv'
    # params.init_teacher = 'checkpoints/DomainNet/painting/ResNet18_baseline/0/80.tar'
    # params.update_teacher = 'none'
    # params.init_student = 'feature'
    # params.momentum = 0.99
    # params.simclr = True
    # params.fixmatch = True
    # params.fixmatch_prior = True
    # params.threshold = 0.5
    # params.fixmatch_teacher = True
    # params.fixmatch_noaug = True
    # params.update_teacher = 'none'
    # params.distribution_align = True
    # params.distribution_m = 1
    # params.fixmatch_gt = True
    # params.fixmatch_anchor = 2
    # params.classcontrast = True
    # params.classcontrast_fn = 'kl'
    # params.pseudomix = True
    # params.pseudomix_fn = 'cutmix'
    # params.ad_align = True
    # params.ad_align_type = 'cada'
    # params.proto_align = True
    # params.rot_align = False
    # params.proto_align = True
    # params.weight_proto = True
    # params.ada_proto = False
    # params.batch_size = 128
    # params.resume = True
    # params.checkpoint = 'checkpoints/DomainNet/clipart/ResNet18_baseline/fixmatch_th0.5_norm-1_prior_l0.2_proto_align_a7_norm-1_crop_val'
    # params.checkpoint = 'checkpoints/DomainNet/clipart/ResNet18_baseline/naive_proto_align_a7_m0.7'
    # params.checkpoint = 'checkpoints/DomainNet/painting/ResNet18_baseline/0'
    # params.save_iter = 20
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

    if params.dataset in ['DomainNet', 'Office-Home']:
        base_folder = os.path.join('../dataset', params.dataset, 'real/base')
        # debug
        # base_folder = os.path.join('../dataset', params.dataset, 'real/novel')

        if params.supervised_align:
            base_folder = [base_folder,
                           os.path.join('../dataset', params.dataset, params.cross_domain, 'base')]
        val_folder = os.path.join('../dataset', params.dataset, 'real', params.split)
        if params.cross_domain:
            val_folder = [os.path.join('../dataset', params.dataset, params.cross_domain, params.split),
                          val_folder]
            if params.reverse_sq:
                val_folder = val_folder[::-1]
            unlabeled_folder = [os.path.join('../dataset', params.dataset, params.cross_domain ,'base'),
                                os.path.join('../dataset', params.dataset, params.cross_domain ,'val')]
                                # f'../dataset/DomainNet/{params.cross_domain}/novel']
            # debug
            # unlabeled_folder = [os.path.join('../dataset', params.dataset, params.cross_domain ,'base')]
    else:
        raise ValueError('unknown dataset')

    image_size = 224
    optimization = params.optimizer

    if params.method in ['baseline', 'baseline++']:
        base_datamgr = SimpleDataManager(image_size, batch_size=params.batch_size)
        base_loader = base_datamgr.get_data_loader(data_folder=base_folder, aug=params.train_aug, drop_last=True,
                                                   fixmatch_trans=params.proto_align, augtype=params.fixmatch_augtype,
                                                   fixmatch_weak=params.fixmatch_weak, fixmatch_anchor=params.fixmatch_anchor)

        few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(image_size, fixmatch_resize=params.proto_align, n_query=15, n_episode=params.n_episode, **few_shot_params)
        val_loader = val_datamgr.get_data_loader(data_folder=val_folder, aug=False, fix_seed=True)

        if params.cross_domain:
            unlabeled_datamgr = SimpleDataManager(image_size, batch_size=params.unlabeled_bs)
            unlabeled_loader = unlabeled_datamgr.get_data_loader(data_folder=unlabeled_folder, aug=params.train_aug,
                                                                 add_label=True,
                                                                 with_idx=True)
            base_loader = {'base': base_loader,
                           'unlabeled': unlabeled_loader}
            if params.fixmatch:
                unlabeled_datamgr = SimpleDataManager(image_size, batch_size=params.fixmatch_bs)
                unlabeled_loader = unlabeled_datamgr.get_data_loader(data_folder=unlabeled_folder,
                                                                     fixmatch_trans=True,
                                                                     augtype=params.fixmatch_augtype,
                                                                     fixmatch_weak=params.fixmatch_weak,
                                                                     fixmatch_anchor=params.fixmatch_anchor,
                                                                     add_label=True,
                                                                     with_idx=True)
                base_loader['fixmatch'] = unlabeled_loader
            if params.classcontrast:
                unlabeled_datamgr = SimpleDataManager(image_size, batch_size=params.classcontrast_bs)
                unlabeled_loader = unlabeled_datamgr.get_data_loader(data_folder=unlabeled_folder,
                                                                     fixmatch_trans=True,
                                                                     augtype=params.classcontrast_augtype,
                                                                     add_label=True,
                                                                     drop_last=True)
                base_loader['classcontrast'] = unlabeled_loader
            if params.simclr:
                simclr_datamgr = SimpleDataManager(image_size, batch_size=params.simclr_bs)
                simclr_loader = simclr_datamgr.get_data_loader(data_folder=unlabeled_folder, simclr_trans=True)
                base_loader['simclr'] = simclr_loader

        if params.bn_align and params.bn_align_mode != 'concat':
            backbone.SimpleBlock.bn = params.bn_align_mode
            backbone.ResNet.bn = params.bn_align_mode
        if params.method == 'baseline':
            model = BaselineTrain(params, model_dict[params.model], params.num_classes)
        elif params.method == 'baseline++':
            model = BaselineTrain(params, model_dict[params.model], params.num_classes, loss_type='dist')
    elif params.method in ['protonet', 'matchingnet', 'relationnet', 'relationnet_softmax', 'maml', 'maml_approx']:
        n_query = max(1, int(15 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

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

    if params.init_teacher:
        tmp = torch.load(params.init_teacher)
        feature = copy.deepcopy(model.feature)
        classifier = copy.deepcopy(model.classifier)
        print(f'init teacher from {params.init_teacher}')
        state = tmp['state']
        if params.bn_align_mode in ['dsbn', 'adabn']:
            keys = model.state_dict().keys()
            for key in keys:
                if 'teacher' not in key:
                    if 'bn1' in key:
                        key2 = key.replace('bn1.', '')
                        state[key] = state[key2]
                    elif 'bn2' in key:
                        key2 = key.replace('bn2.', '')
                        state[key] = state[key2]
        elif params.bn_align_mode == 'dsconv':
            keys = model.state_dict().keys()
            for key in keys:
                if 'teacher' not in key:
                    if 'conv0' in key:
                        key2 = key.replace('conv0.', '')
                        state[key] = state[key2]
        model.load_state_dict(state, strict=False)
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
    # source_diff = []
    # target_acc_fc = []
    # target_acc_proto = []
    # target_diff = []
    # for save_iter in range(0, 40, 10):
    #     checkpoint_dir = params.checkpoint
    #     resume_file = get_resume_file(checkpoint_dir, save_iter)
    #     tmp = torch.load(resume_file)
    #     model.load_state_dict(tmp['state'], strict=False)
    #     model.eval()
        # acc_fc = 0.
        # proto = [0] * model.num_class
        # sample_num = [0] * model.num_class
        # with torch.no_grad():
        #     for x, y in base_loader[0]:
        #         x, y = x.cuda(), y.cuda()
        #         fx = model.feature(x)
        #         score = model.classifier(fx)
        #         pred = score.max(dim=-1)[1]
        #         acc_fc += float((pred==y).sum().item()) / pred.shape[0]
        #         for i in range(fx.shape[0]):
        #             proto[y[i]] += fx[i]
        #             sample_num[y[i]] += 1
        # acc_fc /= len(base_loader[0])
        # for i in range(model.num_class):
        #     proto[i] /= sample_num[i]
        # proto = torch.stack(proto)
        # acc_proto = 0.
        # with torch.no_grad():
        #     for x, y in base_loader[0]:
        #         x, y = x.cuda(), y.cuda()
        #         fx = model.feature(x)
        #         score = - ((fx.unsqueeze(1) - proto.unsqueeze(0)) ** 2).sum(dim=-1)
        #         pred = score.max(dim=-1)[1]
        #         acc_proto += float((pred == y).sum().item()) / pred.shape[0]
        # acc_proto /= len(base_loader[0])
        # source_acc_fc.append(acc_fc)
        # source_acc_proto.append(acc_proto)

        # acc_fc, acc_proto, diff, num = 0., 0., 0., 0.
        # proto = model.target_proto
        # with torch.no_grad():
        #     for x, y, *_ in base_loader['base']:
        #         x, y = x[1].cuda(), y.cuda()
        #         fx = model.feature(x)
        #         score = model.classifier(fx)
        #         pred = score.max(dim=-1)[1]
        #         acc_fc += float((pred == y).sum().item())
        #         score = - ((fx.unsqueeze(1) - proto.unsqueeze(0)) ** 2).sum(dim=-1)
        #         pred2 = score.max(dim=-1)[1]
        #         acc_proto += float((pred2 == y).sum().item())
        #         diff += float((pred2 == pred).sum().item())
        #         num += x.shape[0]
        # acc_fc /= num
        # acc_proto /= num
        # diff /= num
        # source_acc_fc.append(acc_fc)
        # source_acc_proto.append(acc_proto)
        # source_diff.append(diff)
        # print('source_acc_fc:', [f'{x*100:.2f}' for x in source_acc_fc])
        # print('source_acc_proto:', [f'{x*100:.2f}' for x in source_acc_proto])
        # print('source_diff:', [f'{x*100:.2f}' for x in source_diff])
        # print(f'target_acc_fc: {target_acc_fc}')
        # print(f'target_acc_proto: {target_acc_proto}')
        # print(f'target_diff: {target_diff}')
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
    # save_iters = [0, 10]
    # # for mode in ['eval', 'train']:
    # model.train()
    # for save_iter in save_iters:
    #     checkpoint_dir = params.checkpoint
    #     resume_file = get_resume_file(checkpoint_dir, save_iter)
    #     tmp = torch.load(resume_file)
    #     model.load_state_dict(tmp['state'], strict=False)
    #     acc_fc = 0.
    #     num = 0.
    #     with torch.no_grad():
    #         for x, y, *_ in base_loader['unlabeled']:
    #             x, y = x.cuda(), y.cuda()
    #             fx = model.feature(x)
    #             score = model.classifier(fx)
    #             pred = score.max(dim=-1)[1]
    #             confidence = torch.nn.functional.softmax(score, -1).max(-1)[0]
    #             acc_fc += (pred == y).sum().item()
    #             num += pred.shape[0]
    #     acc_fc /= num
    #     target_acc.append(acc_fc)
    #     print(f'target_acc: {target_acc}')
    # exit()

    # mean, var = 0., 0.
    # resume_file = get_resume_file(params.checkpoint, -1)
    # tmp = torch.load(resume_file)
    # model.load_state_dict(tmp['state'], strict=False)
    # model.eval()
    #
    # with torch.no_grad():
    #     for i, (x, y, *_) in enumerate(base_loader['unlabeled']):
    #         model.feature.set_bn_choice('b')
    #         fx = model.feature(x.cuda())
    #         mean_i, var_i = model.feature.get_mean_var()
    #         if i == 0:
    #             mean, var = mean_i, var_i
    #         else:
    #             for j, (avg, cur) in enumerate(zip(mean, mean_i)):
    #                 mean[j] = (avg * i + cur) / (i+1)
    #             for j, (avg, cur) in enumerate(zip(var, var_i)):
    #                 var[j] = (avg * i + cur) / (i+1)
    # torch.save([mean, var], 'startup_u_mean_var.pt')
    # exit()


    # visualize domain distance and class distance
    # import tqdm
    # dists = []
    # in_dists = []
    # target_in_dists = []
    # inter_dists = []
    # target_inter_dists = []
    # ratios = []
    # target_ratios = []
    # model.num_class = 65
    # for save_iter in range(0, 100, 10):
    #     checkpoint_dir = params.checkpoint
    #     resume_file = get_resume_file(checkpoint_dir, save_iter)
    #     tmp = torch.load(resume_file)
    #     model.load_state_dict(tmp['state'], strict=False)
    #     model.eval()
    #
    #     proto = [0] * model.num_class
    #     sample_num = [0] * model.num_class
    #     with torch.no_grad():
    #         for x, y in tqdm.tqdm(base_loader['base']):
    #             xw, xs, y = x[0].cuda(), x[1].cuda(), y.cuda()
    #             fx = model.feature(xw)
    #             for i in range(fx.shape[0]):
    #                 proto[y[i]] += fx[i]
    #                 sample_num[y[i]] += 1
    #     for i in range(model.num_class):
    #         proto[i] /= sample_num[i]
    #     proto = torch.stack(proto)
    #
    #     # in_dist = [0] * model.num_class
    #     # sample_num = [0] * model.num_class
    #     # with torch.no_grad():
    #     #     for x, y in tqdm.tqdm(base_loader['base']):
    #     #         xw, xs, y = x[0].cuda(), x[1].cuda(), y.cuda()
    #     #         fx = model.feature(xw)
    #     #         for i in range(fx.shape[0]):
    #     #             in_dist[y[i]] += ((fx[i] - proto[y[i]]) ** 2).sum().sqrt().item()
    #     #             sample_num[y[i]] += 1
    #     # for i in range(model.num_class):
    #     #     in_dist[i] /= sample_num[i]
    #     # in_dist = sum(in_dist) / model.num_class
    #     # in_dists.append(f'{in_dist:.3f}')
    #     #
    #     # inter_dist = ((proto.unsqueeze(0) - proto.unsqueeze(1)) ** 2).sum(-1).sqrt()
    #     # inter_dist *= (1 - torch.eye(model.num_class).cuda())
    #     # inter_dist = inter_dist.sum() / (model.num_class * (model.num_class - 1))
    #     # inter_dists.append(f'{inter_dist:.3f}')
    #
    #     ratio = 0
    #     num = 0
    #     with torch.no_grad():
    #         for x, y in tqdm.tqdm(base_loader['base']):
    #             xw, xs, y = x[0].cuda(), x[1].cuda(), y.cuda()
    #             fx = model.feature(xw)
    #             d = ((fx.unsqueeze(1) - proto.unsqueeze(0)) ** 2).sum(-1).sqrt()
    #             for i in range(d.shape[0]):
    #                 i_d = d[i, y[i]].item()
    #                 # o_d = (d[i].sum() - i_d) / (model.num_class - 1)
    #                 d[i, y[i]] = 1e6
    #                 o_d = d[i].min().item()
    #                 ratio += i_d / o_d
    #                 num += 1
    #     ratio /= num
    #     ratios.append(f'{ratio:.3f}')
    #
    #     target_proto = [0] * model.num_class
    #     sample_num = [0] * model.num_class
    #     with torch.no_grad():
    #         for x, y, *_ in tqdm.tqdm(base_loader['fixmatch']):
    #             xw, xs, y = x[0].cuda(), x[1].cuda(), y.cuda()
    #             fx = model.feature(xw)
    #             for i in range(fx.shape[0]):
    #                 if y[i] < model.num_class:
    #                     target_proto[y[i]] += fx[i]
    #                     sample_num[y[i]] += 1
    #     target_mean = sum(target_proto) / sum(sample_num)
    #     for i in range(model.num_class):
    #         if sample_num[i] == 0:
    #             target_proto[i] = target_mean
    #         else:
    #             target_proto[i] /= sample_num[i]
    #     target_proto = torch.stack(target_proto)
    #     # target_proto = model.target_proto
    #
    #     # target_in_dist = [0] * model.num_class
    #     # sample_num = [0] * model.num_class
    #     # with torch.no_grad():
    #     #     for x, y, *_ in tqdm.tqdm(base_loader['fixmatch']):
    #     #         xw, xs, y = x[0].cuda(), x[1].cuda(), y.cuda()
    #     #         fx = model.feature(xw)
    #     #         for i in range(fx.shape[0]):
    #     #             target_in_dist[y[i]] += ((fx[i] - target_proto[y[i]]) ** 2).sum().sqrt().item()
    #     #             sample_num[y[i]] += 1
    #     # for i in range(model.num_class):
    #     #     target_in_dist[i] /= sample_num[i]
    #     # target_in_dist = sum(target_in_dist) / model.num_class
    #     # target_in_dists.append(f'{target_in_dist:.3f}')
    #     #
    #     # target_inter_dist = ((target_proto.unsqueeze(0) - target_proto.unsqueeze(1)) ** 2).sum(-1).sqrt()
    #     # target_inter_dist *= (1 - torch.eye(model.num_class).cuda())
    #     # target_inter_dist = target_inter_dist.sum() / (model.num_class * (model.num_class - 1))
    #     # target_inter_dists.append(f'{target_inter_dist:.3f}')
    #
    #     ratio = 0
    #     num = 0
    #     with torch.no_grad():
    #         for x, y, *_ in tqdm.tqdm(base_loader['fixmatch']):
    #             xw, xs, y = x[0].cuda(), x[1].cuda(), y.cuda()
    #             fx = model.feature(xw)
    #             d = ((fx.unsqueeze(1) - target_proto.unsqueeze(0)) ** 2).sum(-1).sqrt()
    #             for i in range(d.shape[0]):
    #                 i_d = d[i, y[i]].item()
    #                 # o_d = (d[i].sum() - i_d) / (model.num_class - 1)
    #                 d[i, y[i]] = 1e6
    #                 o_d = d[i].min().item()
    #                 ratio += i_d / o_d
    #                 num += 1
    #     ratio /= num
    #     target_ratios.append(f'{ratio:.3f}')
    #
    #     dist = ((proto - target_proto) ** 2).sum(-1).sqrt().mean(0).item()
    #
    #     dists.append(f'{dist:.3f}')
    #     print(dists)
    #     # print(in_dists)
    #     # print(target_in_dists)
    #     # print(inter_dists)
    #     # print(target_inter_dists)
    #     print(ratios)
    #     print(target_ratios)
    # exit()

    # t-SNE
    # import matplotlib.pyplot as plt
    # from sklearn.manifold import TSNE
    #
    # def plot_embedding(source_data, source_label, source_proto, target_data, target_label, target_proto, classes, filename):
    #     # x_min, x_max = np.min(data, 0), np.max(data, 0)
    #     # data = (data - x_min) / (x_max - x_min)
    #
    #     fig = plt.figure()
    #     plt.scatter(source_data[:, 0], source_data[:, 1], c=source_label*1.0/source_label.max(), marker='^', cmap='Set2', s=20, alpha=1., linewidths=0.5)
    #     plt.scatter(target_data[:, 0], target_data[:, 1], c=target_label * 1.0 / target_label.max(), marker='v', cmap='Set2', s=20, alpha=1., linewidths=0.5)
    #     plt.scatter(source_proto[:, 0], source_proto[:, 1], c=classes * 1.0 / classes.max(), marker='^', cmap='Set2', s=50, alpha=1., linewidths=1, edgecolors='black')
    #     plt.scatter(target_proto[:, 0], target_proto[:, 1], c=classes * 1.0 / classes.max(), marker='v', cmap='Set2', s=50, alpha=1., linewidths=1, edgecolors='black')
    #     # plt.axis('off')
    #     # plt.savefig('scatter_{:03d}.png'.format(itr), bbox_inches='tight')
    #     # plt.close(f)
    #     # for i in range(data.shape[0]):
    #     #     plt.text(data[i, 0], data[i, 1], str(label[i]),
    #     #              color=plt.cm.Set1(label[i] * 1.0 / label.max()),
    #     #              fontdict={'weight': 'bold', 'size': 9})
    #     # plt.xticks([])
    #     # plt.yticks([])
    #     plt.savefig(filename, bbox_inches='tight')
    #     return fig
    #
    # source_features = []
    # source_labels = []
    # target_features = []
    # target_labels = []
    # for save_iter in range(0, 10, 10):
    #     checkpoint_dir = params.checkpoint
    #     resume_file = get_resume_file(checkpoint_dir, save_iter)
    #     tmp = torch.load(resume_file)
    #     model.load_state_dict(tmp['state'], strict=False)
    #     model.eval()
    #     with torch.no_grad():
    #         for x, y, *_ in base_loader['base']:
    #             x = x[0].cuda()
    #             fx = model.feature(x)
    #             fx = torch.nn.functional.normalize(fx, dim=-1)
    #             source_features.append(fx.cpu())
    #             source_labels.append(y)
    #     source_features = torch.cat(source_features, 0).numpy()
    #     source_labels = torch.cat(source_labels, 0).numpy()
    #     with torch.no_grad():
    #         for x, y, *_ in base_loader['fixmatch']:
    #             x = x[0].cuda()
    #             fx = model.feature(x)
    #             fx = torch.nn.functional.normalize(fx, dim=-1)
    #             target_features.append(fx.cpu())
    #             target_labels.append(y)
    #     target_features = torch.cat(target_features, 0).numpy()
    #     target_labels = torch.cat(target_labels, 0).numpy()
    #     print('get all features!')
    #     pickle.dump((source_features, source_labels, target_features, target_labels), open(f'feature_label_epoch{save_iter}.pkl', 'wb'))
    #     # source_features, source_labels, target_features, target_labels = pickle.load(open(f'feature_label_epoch{save_iter}}.pkl', 'rb'))
    #     idx = source_labels < 7
    #     source_features, source_labels = source_features[idx], source_labels[idx]
    #     idx = target_labels < 7
    #     target_features, target_labels = target_features[idx], target_labels[idx]
    #     features = np.concatenate([source_features, target_features], axis=0)
    #     labels = np.concatenate([source_labels, target_labels], axis=0)
    #     classes = np.unique(labels)
    #     tsne = TSNE(n_components=2, init='pca', random_state=0)
    #     result = tsne.fit_transform(features)
    #     # pickle.dump((result, labels), open('tsne_result.pkl', 'wb'))
    #     # result, labels = pickle.load(open('tsne_result.pkl', 'rb'))
    #     num_source = source_features.shape[0]
    #     source_result, target_result = result[:num_source], result[num_source:]
    #     source_proto, target_proto = [], []
    #     for c in classes:
    #         source_proto.append(source_result[source_labels==c].mean(0))
    #         target_proto.append(target_result[target_labels==c].mean(0))
    #     source_proto, target_proto = np.stack(source_proto), np.stack(target_proto)
    #     fig = plot_embedding(source_result, source_labels, source_proto, target_result, target_labels, target_proto, classes, filename=f'epoch-{save_iter}.png')
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
