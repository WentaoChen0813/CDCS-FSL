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
    for epoch in range(start_epoch,stop_epoch):
        model.train()
        end = time.time()
        loss = model.train_loop(epoch, base_loader,  optimizer ) #model are called by reference, no need to return
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
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

    return model

if __name__=='__main__':
    params = parse_args('train')
    #dubug
    # params.exp = 'painting_real_ad_align'
    # params.ad_align = True
    # params.resume = True
    # params.save_iter = 50
    os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
    if params.seed >= 0:
        np.random.seed(params.seed)
        random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.backends.cudnn.deterministic = True


    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json' 
        val_file   = configs.data_dir['CUB'] + 'val.json' 
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json' 
        val_file   = configs.data_dir['emnist'] + 'val.json'
    elif params.dataset == 'DomainNet':
        base_folder = '/mnt/sdb/wentao/few-shot-learning/dataset/DomainNet/real/base/'
        if params.supervised_align:
            base_folder = [base_folder,
                           f'/mnt/sdb/wentao/few-shot-learning/dataset/DomainNet/{params.cross_domain}/base/']
        val_folder = os.path.join('/mnt/sdb/wentao/few-shot-learning/dataset/DomainNet/real/',
                                   params.split)
        if params.cross_domain:
            val_folder = [os.path.join(f'/mnt/sdb/wentao/few-shot-learning/dataset/DomainNet/{params.cross_domain}/',
                                       params.split),
                          val_folder
                          ]
        if params.cross_domain and params.ad_align > 0:
            unlabeled_folder = [f'/mnt/sdb/wentao/few-shot-learning/dataset/DomainNet/{params.cross_domain}/base',
                                f'/mnt/sdb/wentao/few-shot-learning/dataset/DomainNet/{params.cross_domain}/val',
                                f'/mnt/sdb/wentao/few-shot-learning/dataset/DomainNet/{params.cross_domain}/novel']

    else:
        base_file = configs.data_dir[params.dataset] + 'base.json' 
        val_file   = configs.data_dir[params.dataset] + 'val.json' 
         
    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224

    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'

    optimization = 'Adam'

    if params.stop_epoch == -1: 
        if params.method in ['baseline', 'baseline++'] :
            if params.dataset in ['omniglot', 'cross_char']:
                params.stop_epoch = 5
            elif params.dataset in ['CUB']:
                params.stop_epoch = 200 # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
            elif params.dataset in ['miniImagenet', 'cross']:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 400 #default
        else: #meta-learning methods
            if params.n_shot == 1:
                params.stop_epoch = 600
            elif params.n_shot == 5:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 600 #default
     

    if params.method in ['baseline', 'baseline++'] :
        base_datamgr    = SimpleDataManager(image_size, batch_size = 128)
        base_loader     = base_datamgr.get_data_loader( data_folder=base_folder, aug=params.train_aug)
        few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(image_size, n_query = 15, n_episode=params.n_episode, **few_shot_params)
        val_loader = val_datamgr.get_data_loader(data_folder=val_folder, aug=False)
        if params.cross_domain and params.ad_align > 0:
            unlabeled_datamgr = SimpleDataManager(image_size, batch_size = 128)
            unlabeled_loader = unlabeled_datamgr.get_data_loader(data_folder=unlabeled_folder, aug=params.train_aug,
                                                                 proportion=params.unlabeled_proportion)
            base_loader = [base_loader, unlabeled_loader]
        
        if params.dataset == 'omniglot':
            assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
        if params.dataset == 'cross_char':
            assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'

        if params.method == 'baseline':
            model           = BaselineTrain( model_dict[params.model], params.num_classes,
                                             ad_align=params.ad_align, ad_loss_weight=params.ad_loss_weight)
        elif params.method == 'baseline++':
            model           = BaselineTrain( model_dict[params.model], params.num_classes, loss_type = 'dist',
                                             ad_align=params.ad_align, ad_loss_weight=params.ad_loss_weight)

    elif params.method in ['protonet','matchingnet','relationnet', 'relationnet_softmax', 'maml', 'maml_approx']:
        n_query = max(1, int(15* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
 
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        base_datamgr            = SetDataManager(image_size, n_query = n_query, **train_few_shot_params)
        base_loader             = base_datamgr.get_data_loader( data_folder=base_folder, aug=params.train_aug, fix_seed=False)
         
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 
        val_datamgr             = SetDataManager(image_size, n_query = n_query, n_episode=params.n_episode, **test_few_shot_params)
        val_loader              = val_datamgr.get_data_loader( data_folder=val_folder, aug = False)
        #a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor        

        if params.method == 'protonet':
            model           = ProtoNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'matchingnet':
            model           = MatchingNet( model_dict[params.model], **train_few_shot_params )
        elif params.method in ['relationnet', 'relationnet_softmax']:
            if params.model == 'Conv4': 
                feature_model = backbone.Conv4NP
            elif params.model == 'Conv6': 
                feature_model = backbone.Conv6NP
            elif params.model == 'Conv4S': 
                feature_model = backbone.Conv4SNP
            else:
                feature_model = lambda: model_dict[params.model]( flatten = False )
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

            model           = RelationNet( feature_model, loss_type = loss_type , **train_few_shot_params )
        elif params.method in ['maml' , 'maml_approx']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            model           = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , **train_few_shot_params )
            if params.dataset in ['omniglot', 'cross_char']: #maml use different parameter in omniglot
                model.n_task     = 32
                model.task_update_num = 1
                model.train_lr = 0.1
    else:
       raise ValueError('Unknown method')

    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
    params.checkpoint_dir += f'/{params.exp}'

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    log_dir = params.checkpoint_dir.replace('checkpoints', 'tensorboard')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    params.logger = Logger(log_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method == 'maml' or params.method == 'maml_approx' :
        stop_epoch = params.stop_epoch * model.n_task #maml use multiple tasks in one update 

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir, save_iter=params.save_iter)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])
    elif params.warmup: #We also support warmup from pretrained baseline feature, but we never used in our paper
        baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None: 
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    if not params.test:
        model = train(base_loader, val_loader,  model, optimization, start_epoch, stop_epoch, params)
    else:
        test(val_loader, model, params)