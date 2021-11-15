import numpy as np
import os
import glob
import argparse
import backbone

model_dict = dict(
            Conv4 = backbone.Conv4,
            Conv4S = backbone.Conv4S,
            Conv6 = backbone.Conv6,
            ResNet10 = backbone.ResNet10,
            ResNet18 = backbone.ResNet18,
            ResNet34 = backbone.ResNet34,
            ResNet50 = backbone.ResNet50,
            ResNet101 = backbone.ResNet101) 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp'         , default='debug'     , help='experiment name')
    parser.add_argument('--gpu'         , default='0')
    parser.add_argument('--dataset'     , default='DomainNet' , help='DomainNet/Office-Home')
    parser.add_argument('--cross_domain', default='painting'  , help='unlabeled target domain')
    parser.add_argument('--model'       , default='ResNet18'  , help='backbone model: Conv{4|6} / ResNet{10|18|34|50|101}')
    parser.add_argument('--method'      , default='baseline'  , help='baseline/baseline++/protonet/deepemd/metaoptnet/matchingnet/relationnet{_softmax}/maml{_approx}')
    parser.add_argument('--train_n_way' , default=5, type=int , help='class num to classify for training')
    parser.add_argument('--test_n_way'  , default=5, type=int , help='class num to classify for testing (validation)')
    parser.add_argument('--n_shot'      , default=5, type=int , help='number of labeled data in each class')
    parser.add_argument('--train_n_shot', default=-1, type=int, help='number of labeled data in each class, same as n_shot by default')
    parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ')

    # train
    parser.add_argument('--seed'        , default=0, type=int)
    parser.add_argument('--optimizer'   , default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--lr'          , default=1e-3, type=float)
    parser.add_argument('--batch_size'  , default=128, type=int, help='batch size of source data')
    parser.add_argument('--unlabeled_bs', default=128, type=int, help='batch size of target data')
    parser.add_argument('--init_teacher', default='')
    parser.add_argument('--init_model'  , default='')
    parser.add_argument('--ts_align'    , action='store_true'  , help='target-to-source alignment')
    parser.add_argument('--ts_align_tau', default=10, type=float)
    parser.add_argument('--st_align'    , action='store_true'  , help='source-to-target alignment')
    parser.add_argument('--prototype_m' , default=0.1, type=float)
    parser.add_argument('--st_align_tau', default=4, type=float)
    parser.add_argument('--mix_lambda'  , default=0.2, type=float)
    parser.add_argument('--aug_type'    , default='none', type=str, choices=['strong', 'weak', 'none'])
    parser.add_argument('--threshold'   , default=0.5, type=float)
    parser.add_argument('--num_classes' , default=228, type=int, help='total number of classes in softmax, only used in baseline')
    parser.add_argument('--save_freq'   , default=10, type=int , help='save frequency')
    parser.add_argument('--start_epoch' , default=0, type=int  , help ='starting epoch')
    parser.add_argument('--stop_epoch'  , default=50, type=int , help ='stopping epoch')
    # test
    parser.add_argument('--test'        , action='store_true'  , help='only test the model')
    parser.add_argument('--test_freq'   , default=1, type=int  , help='testing frequency during training')
    parser.add_argument('--reverse_sq'  , action='store_true'  , help='if set true, the support set and query set are from the source domain and target domain, respectively')
    parser.add_argument('--n_episode'   , default=600, type=int, help='number of testing episodes')
    parser.add_argument('--split'       , default='val',         help='base/val/novel')
    parser.add_argument('--resume'      , action='store_true'  , help='continue from previous trained model')
    parser.add_argument('--checkpoint'  , default='')
    parser.add_argument('--save_iter'   , default=-1, type=int , help='saved feature from the model trained in x epoch, use the best model if x is -1')

    return parser.parse_args()


def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir, save_iter=None):
    if save_iter is None:
        filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
        if len(filelist) == 0:
            return None
        filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
        epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
        max_epoch = np.max(epochs)
        resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
        return resume_file
    elif save_iter == -1:
        return get_best_file(checkpoint_dir)
    else:
        return get_assigned_file(checkpoint_dir, save_iter)


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
