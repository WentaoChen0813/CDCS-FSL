import backbone
import utils
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type = 'softmax', ad_align=False, ad_loss_weight=0.001):
        super(BaselineTrain, self).__init__()
        self.feature    = model_func()
        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist': #Baseline ++
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)
        self.loss_type = loss_type  #'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.DBval = False #only set True for CUB dataset, see issue #31
        self.ad_align = ad_align
        self.ad_loss_weight = ad_loss_weight
        if ad_align:
            feat_dim = self.feature.final_feat_dim
            self.discriminator = nn.Sequential(
                nn.Linear(feat_dim, feat_dim//2),
                nn.LeakyReLU(0.2),
                nn.Linear(feat_dim//2, feat_dim//4),
                nn.LeakyReLU(0.2),
                nn.Linear(feat_dim//4, 2)
            )


    def forward(self,x):
        x    = x.cuda()
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def forward_loss(self, x, y):
        x = x.cuda()
        feature = self.feature(x)
        scores = self.classifier(feature)
        y = y.cuda()
        return self.loss_fn(scores, y ), feature


    def discriminator_loss(self, x, y, is_feature=False):
        if not is_feature:
            x = x.cuda()
            x = self.feature(x)
        pred = self.discriminator(x)
        y = torch.ones(pred.shape[0], dtype=torch.long).cuda() * y
        return self.loss_fn(pred, y), x
    
    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss=0
        avg_ad_loss = 0
        avg_discriminator_loss = 0

        if self.ad_align:
            train_loader, unlabeled_loader = train_loader
            n_unlabeled = len(unlabeled_loader)
            optimizer, discriminator_optim = optimizer
        for i, (x,y) in enumerate(train_loader):
            loss, fx = self.forward_loss(x, y)
            avg_loss = avg_loss + loss.item()

            if self.ad_align:
                if i % n_unlabeled == 0:
                    unlabeled_iter = iter(unlabeled_loader)
                ux, _ = next(unlabeled_iter)
                ad_loss, fux = self.discriminator_loss(ux, 1)
                ad_loss *= self.ad_loss_weight
                avg_ad_loss += ad_loss.item()
                loss += avg_ad_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.ad_align:
                real_loss, _ = self.discriminator_loss(fx.detach(), 1, is_feature=True)
                fake_loss, _ = self.discriminator_loss(fux.detach(), 0, is_feature=True)
                discriminator_loss = (real_loss + fake_loss) / 2
                avg_discriminator_loss += discriminator_loss.item()

                discriminator_optim.zero_grad()
                discriminator_loss.backward()
                discriminator_optim.step()

            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                if not self.ad_align:
                    print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)  ))
                else:
                    print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Ad_loss {:f} | Discriminator_loss {:f}'.format(
                        epoch, i, len(train_loader), avg_loss/(i+1), avg_ad_loss/(i+1), avg_discriminator_loss/(i+1)))
        if not self.ad_align:
            return avg_loss / (i+1)
        else:
            return {'loss': avg_loss/(i+1),
                    'ad_loss': avg_ad_loss/(i+1),
                    'discriminator_loss': avg_discriminator_loss/(i+1)}
                     
    def test_loop(self, epoch, val_loader, params):
        # if self.DBval:
        #     return self.analysis_loop(val_loader)
        # else:
        #     return -1   #no validation, just save model during iteration
        accs = []
        with torch.no_grad():
            for img, label in tqdm(val_loader):
                n_way = params.test_n_way
                n_shot = params.n_shot
                n_query = img.shape[1] - n_shot

                support_label = torch.arange(n_way).unsqueeze(1).repeat(1, n_shot).view(-1).numpy()
                query_label = torch.arange(n_way).unsqueeze(1).repeat(1, n_query).view(-1).numpy()

                img = img.cuda()
                img = img.view(-1, *img.shape[2:])
                features = self.feature(img)
                features = F.normalize(features, dim=1)
                features = features.view(n_way, n_shot+n_query, -1)
                support_feature = features[:, :n_shot].detach().cpu().numpy().reshape(n_way*n_shot, -1)
                query_feature = features[:, n_shot:].detach().cpu().numpy().reshape(n_way*n_query, -1)

                clf = LogisticRegression(penalty='l2',
                                         random_state=0,
                                         C=1.0,
                                         solver='lbfgs',
                                         max_iter=1000,
                                         multi_class='multinomial')
                clf.fit(support_feature, support_label)
                query_pred = clf.predict(query_feature)
                acc = np.equal(query_pred, query_label).sum() / query_label.shape[0]
                accs.append(acc*100)
        acc_mean = np.mean(accs)
        acc_std = np.std(accs)
        print('Epoch %d, Test Acc = %4.2f%% +- %4.2f%%' % (epoch, acc_mean, 1.96 * acc_std / np.sqrt(len(accs))))
        return acc_mean

    def analysis_loop(self, val_loader, record = None):
        class_file  = {}
        for i, (x,y) in enumerate(val_loader):
            x = x.cuda()
            x_var = Variable(x)
            feats = self.feature.forward(x_var).data.cpu().numpy()
            labels = y.cpu().numpy()
            for f, l in zip(feats, labels):
                if l not in class_file.keys():
                    class_file[l] = []
                class_file[l].append(f)
       
        for cl in class_file:
            class_file[cl] = np.array(class_file[cl])
        
        DB = DBindex(class_file)
        print('DB index = %4.2f' %(DB))
        return 1/DB #DB index: the lower the better

def DBindex(cl_data_file):
    #For the definition Davis Bouldin index (DBindex), see https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
    #DB index present the intra-class variation of the data
    #As baseline/baseline++ do not train few-shot classifier in training, this is an alternative metric to evaluate the validation set
    #Emperically, this only works for CUB dataset but not for miniImagenet dataset

    class_list = cl_data_file.keys()
    cl_num= len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append( np.mean(cl_data_file[cl], axis = 0) )
        stds.append( np.sqrt(np.mean( np.sum(np.square( cl_data_file[cl] - cl_means[-1]), axis = 1))))

    mu_i = np.tile( np.expand_dims( np.array(cl_means), axis = 0), (len(class_list),1,1) )
    mu_j = np.transpose(mu_i,(1,0,2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis = 2))
    
    for i in range(cl_num):
        DBs.append( np.max([ (stds[i]+ stds[j])/mdists[i,j]  for j in range(cl_num) if j != i ]) )
    return np.mean(DBs)
