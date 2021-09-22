import numpy as np
import torch
import torch.optim
import time
import os
import random
import copy

from Logger import Logger
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.protonet import ProtoNet
from methods.deep_emd import DeepEMD
from methods.meta_optnet import MetaOptNet
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 0
    for epoch in range(start_epoch,stop_epoch):
        model.train()
        end = time.time()
        loss = model.train_loop(epoch, base_loader, optimizer, params)
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
            if acc > max_acc :
                print("best model! save...")
                max_acc = acc
                outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

    return model


if __name__ == '__main__':
    params = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
    if params.seed >= 0:
        np.random.seed(params.seed)
        random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.backends.cudnn.deterministic = True

    if params.dataset in ['DomainNet', 'Office-Home']:
        base_folder = os.path.join('dataset', params.dataset, 'real/base')
        val_folder = os.path.join('dataset', params.dataset, 'real', params.split)
        if params.cross_domain:
            val_folder = [os.path.join('dataset', params.dataset, params.cross_domain, params.split),
                          val_folder]
            if params.reverse_sq:
                val_folder = val_folder[::-1]
            unlabeled_folder = [os.path.join('dataset', params.dataset, params.cross_domain ,'base'),
                                os.path.join('dataset', params.dataset, params.cross_domain ,'val')]
    else:
        raise ValueError('unknown dataset')

    image_size = 224
    optimization = params.optimizer

    if params.method in ['baseline', 'baseline++']:
        base_datamgr = SimpleDataManager(image_size, batch_size=params.batch_size)
        base_loader = base_datamgr.get_data_loader(data_folder=base_folder, aug=params.train_aug,
                                                   aug_type=params.aug_type,
                                                   drop_last=True)

        few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(image_size, aug_type=params.aug_type, n_query=15, n_episode=params.n_episode, **few_shot_params)
        val_loader = val_datamgr.get_data_loader(data_folder=val_folder, aug=False, fix_seed=True)

        if params.cross_domain:
            unlabeled_datamgr = SimpleDataManager(image_size, batch_size=params.unlabeled_bs)
            unlabeled_loader = unlabeled_datamgr.get_data_loader(data_folder=unlabeled_folder, aug=params.train_aug,
                                                                 aug_type=params.aug_type,
                                                                 with_idx=True)
            base_loader = {'base': base_loader,
                           'unlabeled': unlabeled_loader}
        if params.method == 'baseline':
            model = BaselineTrain(params, model_dict[params.model], params.num_classes)
        elif params.method == 'baseline++':
            model = BaselineTrain(params, model_dict[params.model], params.num_classes, loss_type='dist')
    elif params.method in ['protonet', 'deepemd', 'metaoptnet', 'matchingnet', 'relationnet', 'relationnet_softmax', 'maml', 'maml_approx']:
        n_query = max(1, int(15 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

        if params.train_n_shot == -1:
            params.train_n_shot = params.n_shot
        train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.train_n_shot)
        base_datamgr = SetDataManager(image_size, n_query=n_query, **train_few_shot_params)
        base_loader = base_datamgr.get_data_loader(data_folder=base_folder, aug=params.train_aug, fix_seed=False)

        test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(image_size, n_query=15, n_episode=params.n_episode, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(data_folder=val_folder, aug=False, fix_seed=True)
        # a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor

        if params.method == 'protonet':
            model = ProtoNet(model_dict[params.model], **train_few_shot_params)
        elif params.method == 'deepemd':
            model = DeepEMD(model_dict[params.model], **train_few_shot_params)
        elif params.method == 'metaoptnet':
            model = MetaOptNet(model_dict[params.model], **test_few_shot_params)
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

    params.checkpoint_dir = 'checkpoints/%s/%s/%s_%s' % (params.dataset, params.cross_domain, params.model, params.method)
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
        model.load_state_dict(state, strict=False)
        model.init_teacher()
        model.feature = feature
        model.classifier = classifier

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

    if not params.test:
        model = train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params)
    else:
        test(val_loader, model, params)
