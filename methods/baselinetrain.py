import backbone
from data.dataset import PseudoPairedSetDataset, EpisodicBatchSampler
from methods.protonet import euclidean_dist
from data.datamgr import FixMatchTransform
import utils
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision


class BaselineTrain(nn.Module):
    def __init__(self, params, model_func, num_class, loss_type='softmax', ad_align=False, ad_loss_weight=0.001,
                 pseudo_align=False, momentum=0.6, threshold=0.9, proto_align=False, ada_proto=False, rot_align=False,
                 scale=2):
        super(BaselineTrain, self).__init__()
        self.params = params
        self.feature = model_func()
        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist':  # Baseline ++
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class, scale)
        elif loss_type == 'euclidean':
            self.classifier = backbone.protoLinear(self.feature.final_feat_dim, num_class)
        self.loss_type = loss_type  # 'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.DBval = False  # only set True for CUB dataset, see issue #31
        self.ad_align = ad_align
        self.ad_loss_weight = ad_loss_weight
        if ad_align:
            feat_dim = self.feature.final_feat_dim
            self.discriminator = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(feat_dim // 2, feat_dim // 4),
                nn.LeakyReLU(0.2),
                nn.Linear(feat_dim // 4, 2)
            )
        self.pseudo_align = pseudo_align
        self.proto_align = proto_align
        if pseudo_align or proto_align or params.pseudomix:
            self.momentum = momentum
            self.threshold = threshold
            self.teacher_feature = model_func()
            if loss_type == 'softmax':
                self.teacher_classifier = nn.Linear(self.feature.final_feat_dim, num_class)
                self.teacher_classifier.bias.data.fill_(0)
            elif loss_type == 'dist':  # Baseline ++
                self.teacher_classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)
            elif loss_type == 'euclidean':
                self.teacher_classifier = backbone.protoLinear(self.feature.final_feat_dim, num_class)
            self.init_teacher()
        self.ada_proto = ada_proto
        if ada_proto:
            self.weight_predictor = torch.nn.Sequential(
                nn.Linear(num_class, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        self.rot_align = rot_align
        if rot_align:
            feat_dim = self.feature.final_feat_dim
            self.rot_classifier = nn.Linear(feat_dim, 4)
        if params.simclr:
            self.projection_head = nn.Sequential(
                nn.Linear(self.feature.final_feat_dim, self.feature.final_feat_dim),
                nn.ReLU(),
                nn.Linear(self.feature.final_feat_dim, 128)
            )
        if params.fixmatch and params.distribution_align:
            self.register_buffer('pred_distribution', torch.ones(self.num_class) / self.num_class)

    def init_teacher(self):
        for param_t, param_s in zip(self.teacher_feature.parameters(), self.feature.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False
        for param_t, param_s in zip(self.teacher_classifier.parameters(), self.classifier.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False

    def update_teacher(self):
        for param_t, param_s in zip(self.teacher_feature.parameters(), self.feature.parameters()):
            param_t.data = self.momentum * param_t.data + (1 - self.momentum) * param_s.data
        for param_t, param_s in zip(self.teacher_classifier.parameters(), self.classifier.parameters()):
            param_t.data = self.momentum * param_t.data + (1 - self.momentum) * param_s.data

    def forward(self, x):
        x = x.cuda()
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)
        return scores

    def teacher_forward(self, x):
        x = x.cuda()
        x = self.teacher_feature(x)
        x = self.teacher_classifier(x)
        x = F.softmax(x, dim=-1)
        max_prob, pred = x.max(dim=-1)
        return max_prob.cpu(), pred.cpu(), x.cpu()

    def forward_loss(self, x, y):
        x = x.cuda()
        feature = self.feature(x)
        scores = self.classifier(feature)
        y = y.cuda()
        if len(y.shape) > 1:
            pred = F.log_softmax(scores, -1)
            loss = F.kl_div(pred, y, reduction='batchmean')
        else:
            loss = self.loss_fn(scores, y)
        return loss, feature

    def discriminator_loss(self, x, y, is_feature=False):
        if not is_feature:
            x = x.cuda()
            x = self.feature(x)
        pred = self.discriminator(x)
        y = torch.ones(pred.shape[0], dtype=torch.long).cuda() * y
        return self.loss_fn(pred, y), x

    def get_pseudo_samples(self, train_loader, params):
        if not params.soft_label:
            with torch.no_grad():
                train_loader, unlabeled_loader = train_loader
                selected_y = []
                selected_idx = []
                for x, _, idx in unlabeled_loader:
                    prob, pred, _ = self.teacher_forward(x)
                    selected_idx.append(idx[prob > self.threshold])
                    selected_y.append(pred[prob > self.threshold])
                selected_idx = torch.cat(selected_idx).detach().cpu().numpy()
                selected_y = torch.cat(selected_y).detach().cpu().numpy()

                class NewDataset:
                    def __init__(self, dataset, label):
                        self.dataset = dataset
                        self.label = label

                    def __getitem__(self, index):
                        data, *_ = self.dataset[index]
                        label = self.label[index]
                        label = -label - 1
                        return data, label

                    def __len__(self):
                        return len(self.dataset)

                if len(selected_y) > 0:
                    n_pseudo = len(selected_y)
                    n_total = len(unlabeled_loader.dataset)
                    print(f'Select {n_pseudo} ({100.0 * n_pseudo / n_total:.2f}%) pesudo samples')
                    pseudo_dataset = torch.utils.data.Subset(unlabeled_loader.dataset, selected_idx)
                    pseudo_dataset = NewDataset(pseudo_dataset, selected_y)
                    labeled_dataset = train_loader.dataset
                    new_dataset = torch.utils.data.ConcatDataset([labeled_dataset, pseudo_dataset])
                    train_loader = torch.utils.data.DataLoader(new_dataset,
                                                               batch_size=train_loader.batch_size,
                                                               shuffle=True,
                                                               num_workers=12,
                                                               drop_last=True)
        if params.soft_label:
            with torch.no_grad():
                train_loader, unlabeled_loader = train_loader
                selected_idx = []
                soft_labels = []
                for x, _, idx in unlabeled_loader:
                    prob, _, soft_label = self.teacher_forward(x)
                    selected_idx.append(idx[prob > self.threshold])
                    soft_labels.append(soft_label[prob > self.threshold])
                selected_idx = torch.cat(selected_idx).detach().cpu().numpy()
                soft_labels = torch.cat(soft_labels).detach().cpu()

                class ToOneHot:
                    def __init__(self, dataset, n_class):
                        self.dataset = dataset
                        self.n_class = n_class

                    def __getitem__(self, index):
                        data, label, *_ = self.dataset[index]
                        label = torch.tensor(label) if not isinstance(label, torch.Tensor) else label
                        label = F.one_hot(label, self.n_class).to(torch.float)
                        return data, label

                    def __len__(self):
                        return len(self.dataset)

                class AddSoftLabel:
                    def __init__(self, dataset, soft_labels):
                        self.dataset = dataset
                        self.soft_labels = soft_labels

                    def __getitem__(self, index):
                        data, *_ = self.dataset[index]
                        soft_label = self.soft_labels[index]
                        return data, soft_label

                    def __len__(self):
                        return len(self.dataset)

                if len(selected_idx) > 0:
                    n_pseudo = len(selected_idx)
                    n_total = len(unlabeled_loader.dataset)
                    print(f'Select {n_pseudo} ({100.0 * n_pseudo / n_total:.2f}%) pesudo samples')
                    if params.target_trans == 'none':
                        pseudo_dataset = torch.utils.data.Subset(unlabeled_loader.dataset, selected_idx)
                    elif params.target_trans == 'randaug':
                        pseudo_dataset = copy.deepcopy(unlabeled_loader.dataset)
                        trans = FixMatchTransform(has_weak=False)
                        for i in range(len(pseudo_dataset.dataset.datasets)):
                            pseudo_dataset.dataset.datasets[i].transform = trans
                        pseudo_dataset = torch.utils.data.Subset(pseudo_dataset, selected_idx)
                    pseudo_dataset = AddSoftLabel(pseudo_dataset, soft_labels)
                    labeled_dataset = train_loader.dataset
                    labeled_dataset = ToOneHot(labeled_dataset, self.num_class)
                    new_dataset = torch.utils.data.ConcatDataset([labeled_dataset, pseudo_dataset])
                    train_loader = torch.utils.data.DataLoader(new_dataset,
                                                               batch_size=train_loader.batch_size,
                                                               shuffle=True,
                                                               num_workers=12,
                                                               drop_last=True)
        return train_loader, unlabeled_loader

    def get_pseudo_loader(self, unlabeled_loader):
        with torch.no_grad():
            selected_y = []
            selected_idx = []
            for x, _, idx in unlabeled_loader:
                prob, pred, _ = self.teacher_forward(x)
                selected_idx.append(idx[prob > self.threshold])
                selected_y.append(pred[prob > self.threshold])
            selected_idx = torch.cat(selected_idx).detach().cpu().numpy()
            selected_y = torch.cat(selected_y).detach().cpu().numpy()

            class NewDataset:
                def __init__(self, dataset, label):
                    self.dataset = dataset
                    self.label = label

                def __getitem__(self, index):
                    data, *_ = self.dataset[index]
                    label = self.label[index]
                    return data, label

                def __len__(self):
                    return len(self.dataset)

            if len(selected_y) > 0:
                n_pseudo = len(selected_y)
                n_total = len(unlabeled_loader.dataset)
                print(f'Select {n_pseudo} ({100.0 * n_pseudo / n_total:.2f}%) pesudo samples')
                pseudo_dataset = torch.utils.data.Subset(unlabeled_loader.dataset, selected_idx)
                pseudo_dataset = NewDataset(pseudo_dataset, selected_y)
                new_loader =  torch.utils.data.DataLoader(pseudo_dataset,
                                                           batch_size=unlabeled_loader.batch_size,
                                                           shuffle=True,
                                                           num_workers=12,
                                                           drop_last=True)
                return new_loader

    def get_pseudo_paired_samples(self, train_loader, unlabeled_loader, params):
        with torch.no_grad():
            selected_idx = []
            selected_pred = []
            if self.rot_align:
                unlabeled_loader.dataset.rot = False
            for x, y, idx in unlabeled_loader:
                if params.gt_proto:
                    selected = y < self.num_class
                    selected_idx.append(idx[selected])
                    selected_pred.append(y[selected])
                else:
                    prob, pred, _ = self.teacher_forward(x)
                    selected = prob > self.threshold
                    selected_idx.append(idx[selected])
                    selected_pred.append(pred[selected])
            selected_idx = torch.cat(selected_idx).detach().cpu().numpy()
            selected_pred = torch.cat(selected_pred).detach().cpu().numpy()
            train_dataset = train_loader.dataset
            unlabeled_dataset = unlabeled_loader.dataset
            if self.rot_align:
                train_dataset = train_dataset.dataset
                unlabeled_dataset = unlabeled_dataset.dataset
            paired_dataset = PseudoPairedSetDataset(train_dataset, unlabeled_dataset,
                                                    selected_idx, selected_pred, n_shot=params.n_shot, n_query=15)
            sampler = EpisodicBatchSampler(len(paired_dataset), params.train_n_way, n_episodes=10000, fix_seed=False)
            data_loader_params = dict(batch_sampler=sampler, num_workers=12, pin_memory=False)
            paired_loader = torch.utils.data.DataLoader(paired_dataset, **data_loader_params)
            n_pseudo = len(selected_pred)
            n_total = len(unlabeled_loader.dataset)
            n_class = len(np.unique(selected_pred))
            print(f'Select {n_pseudo} ({100.0 * n_pseudo / n_total:.2f}%) pesudo samples, {n_class} pseudo classes')
            if self.rot_align:
                unlabeled_loader.dataset.rot = True
            return paired_loader

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.params.simclr_bs) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.params.simclr_t
        return logits, labels

    def mixup(self, x1, x2, y1, y2, alpha):
        beta = torch.distributions.beta.Beta(alpha, alpha)
        lam = beta.sample([x1.shape[0]]).to(device=x1.device)
        lam = torch.max(lam, 1. - lam)
        lam_expanded = lam.view([-1] + [1] * (x1.dim() - 1))
        x = lam_expanded * x1 + (1. - lam_expanded) * x2
        y1 = F.one_hot(y1, self.num_class).float()
        y2 = F.one_hot(y2, self.num_class).float()
        y = lam.unsqueeze(-1) * y1 + (1 - lam.unsqueeze(-1)) * y2
        return x, y

    def train_loop(self, epoch, base_loader, optimizer, params=None):
        print_freq = 10
        avg_loss = 0
        avg_ad_loss = 0
        avg_discriminator_loss = 0
        avg_proto_loss = 0
        avg_rot_loss = 0
        avg_simclr_loss = 0
        avg_fixmatch_loss = 0
        avg_classcontrast_loss = 0

        if self.ad_align or self.pseudo_align or self.proto_align or self.params.fixmatch or self.params.classcontrast or self.params.pseudomix:
            if self.params.simclr:
                train_loader, unlabeled_loader, simclr_loader = base_loader
                simclr_iter = iter(simclr_loader)
            elif (self.params.pseudo_align or self.params.pseudomix) and self.params.classcontrast:
                train_loader, pseudo_loader, unlabeled_loader = base_loader
            else:
                train_loader, unlabeled_loader = base_loader
            unlabeled_iter = iter(unlabeled_loader)

        if params.pseudomix and epoch == 0:
            pseudo_loader = self.get_pseudo_loader(pseudo_loader)
            pseudo_iter = iter(pseudo_loader)
            base_loader[1] = pseudo_loader

        if self.ad_align:
            optimizer, discriminator_optim = optimizer
        if self.proto_align:
            paired_loader = self.get_pseudo_paired_samples(train_loader, unlabeled_loader, params)
            paired_iter = iter(paired_loader)

        for i, (x, y) in enumerate(train_loader):
            if self.rot_align:
                y, rot = y
            if self.pseudo_align:
                if not params.soft_label:
                    pseudo_idx = y < 0
                    real_idx = y >= 0
                    y[pseudo_idx] = -(y[pseudo_idx] + 1)
                    n_real = sum(real_idx).item()
            if self.params.pseudomix:
                try:
                    ux, uy, *_ = next(pseudo_iter)
                except:
                    pseudo_iter = iter(pseudo_loader)
                    ux, uy, *_ = next(pseudo_iter)
                x, y = self.mixup(x.cuda(), ux.cuda(), y.cuda(), uy.cuda(), self.params.pseudomix_alpha)
                fx = self.feature(x)
                logit = self.classifier(fx)
                loss = F.kl_div(F.log_softmax(logit, -1), y, reduction='batchmean')
            else:
                loss, fx = self.forward_loss(x, y)
            avg_loss = avg_loss + loss.item()

            # if self.rot_align:
            # rot_loss = self.loss_fn(self.rot_classifier(fx), rot.cuda())

            if self.ad_align:
                try:
                    ux, y, *_ = next(unlabeled_iter)
                except:
                    unlabeled_iter = iter(unlabeled_loader)
                    ux, y, *_ = next(unlabeled_iter)
                if isinstance(ux, list):
                    ux = ux[0]
                if self.pseudo_align:
                    ux = ux[:n_real]
                ad_loss, fux = self.discriminator_loss(ux, 1)
                avg_ad_loss += ad_loss.item()
                ad_loss *= self.ad_loss_weight
                loss += ad_loss

                if self.rot_align:
                    _, rot = y
                    rot_loss = self.loss_fn(self.rot_classifier(fux), rot.cuda())
                    avg_rot_loss += rot_loss.item()
                    loss += rot_loss

            if self.proto_align:
                sqx, _ = next(paired_iter)
                n_way = params.train_n_way
                n_shot = params.n_shot
                n_query = sqx.shape[1] - n_shot
                sqx = sqx.view(-1, *sqx.shape[2:]).cuda()
                sqfx = self.feature(sqx)
                sqfx = sqfx.view(n_way, n_shot + n_query, -1)
                sfx, qfx = sqfx[:, :n_shot], sqfx[:, n_shot:]
                if self.ada_proto:
                    sx = sqx.view(n_way, n_shot + n_query, *sqx.shape[1:])[:, :n_shot]
                    sx = sx.contiguous().view(n_way * n_shot, *sx.shape[2:])
                    with torch.no_grad():
                        sx = sx.cuda()
                        sx = self.teacher_feature(sx)
                        logit = self.teacher_classifier(sx)
                        prob = torch.softmax(logit, dim=-1)
                        max_logit, pred = logit.max(dim=-1)
                    weight = self.weight_predictor(prob)
                    weight += max_logit.unsqueeze(-1)
                    logit = torch.scatter(logit, dim=-1, index=pred.unsqueeze(-1), src=weight)
                    prob = torch.softmax(logit, -1)
                    weight = torch.gather(prob, dim=-1, index=pred.unsqueeze(-1))
                    weight = weight.view(n_way, n_shot)
                    weight = weight / (weight.sum(dim=-1, keepdim=True) + 1e-6)
                    sfx = sfx.view(n_way, n_shot, -1)
                    proto = (sfx * weight.unsqueeze(-1)).sum(dim=1)
                elif params.weight_proto:
                    sx = sqx.view(n_way, n_shot + n_query, *sqx.shape[1:])[:, :n_shot]
                    sx = sx.contiguous().view(n_way * n_shot, *sx.shape[2:])
                    with torch.no_grad():
                        weight, *_ = self.teacher_forward(sx)
                        weight = weight.view(n_way, n_shot).cuda()
                        weight = weight / (weight.sum(dim=-1, keepdim=True) + 1e-6)
                    proto = (sfx * weight.unsqueeze(-1)).sum(dim=1)
                else:
                    proto = sfx.mean(dim=1)
                qfx = qfx.contiguous().view(n_way * n_query, -1)
                score = -euclidean_dist(qfx, proto)
                qy = torch.arange(n_way).unsqueeze(-1).repeat(1, n_query).view(-1).cuda()
                proto_loss = self.loss_fn(score, qy)
                avg_proto_loss += proto_loss.item()
                loss += proto_loss

            if self.params.simclr:
                try:
                    ux, _ = next(simclr_iter)
                except:
                    simclr_iter = iter(simclr_loader)
                    ux, _ = next(simclr_iter)
                if self.params.simclr_prob:
                    ux0, ux1 = ux[0].cuda(), ux[1].cuda()
                    with torch.no_grad():
                        prob0 = F.softmax(self.classifier(self.feature(ux0)), dim=-1)
                        mask = prob0.max(dim=-1)[0] > self.params.classcontrast_th
                    logit1 = self.classifier(self.feature(ux1))
                    logprob1 = F.log_softmax(logit1, -1)
                    simclr_loss = F.kl_div(logprob1, prob0, reduction='none').sum(-1)
                    simclr_loss = (simclr_loss * mask).mean()
                else:
                    ux = torch.cat(ux, dim=0).cuda()
                    fux = self.feature(ux)
                    fux = self.projection_head(fux)
                    logits, labels = self.info_nce_loss(fux)
                    simclr_loss = self.loss_fn(logits, labels)
                avg_simclr_loss += simclr_loss.item()
                loss += simclr_loss

            if self.params.fixmatch:
                try:
                    ux, uy, *_ = next(unlabeled_iter)
                except:
                    unlabeled_iter = iter(unlabeled_loader)
                    ux, uy, *_ = next(unlabeled_iter)
                ux = [x.cuda() for x in ux]
                if self.params.fixmatch_anchor == 1:
                    uxs = ux[1]
                else:
                    uxs = torch.cat(ux[1:])
                if not self.params.fixmatch_gt:
                    fuxw = self.feature(ux[0])
                    scorew = self.classifier(fuxw)
                    with torch.no_grad():
                        pred = F.softmax(scorew, dim=-1)
                        if self.params.distribution_align:
                            with torch.no_grad():
                                self.pred_distribution = self.params.distribution_m * self.pred_distribution + \
                                                         (1 - self.params.distribution_m) * pred.mean(0)
                                pred = pred * (1. / self.num_class) / (self.pred_distribution.unsqueeze(0) + 1e-6)
                                pred = pred / pred.sum(dim=-1, keepdim=True)
                        pseudo_label = pred.max(dim=-1)[1].detach()
                        confidence = pred.max(dim=-1)[0].detach()
                        mask = confidence.ge(self.params.threshold).float()
                        if self.params.fixmatch_anchor > 1:
                            pseudo_label = pseudo_label.repeat(self.params.fixmatch_anchor)
                            mask = mask.repeat(self.params.fixmatch_anchor)
                    score = self.classifier(self.feature(uxs))
                    fixmatch_loss = (F.cross_entropy(score, pseudo_label, reduction='none') * mask).mean()
                else:
                    if self.params.fixmatch_anchor > 1:
                        uy = uy.repreat(self.params.fixmatch_anchor)
                    uy = uy.cuda()
                    select = uy < self.num_class
                    uy = uy[select]
                    score = self.classifier(self.feature(uxs))
                    score = score[select]
                    fixmatch_loss = F.cross_entropy(score, uy)
                avg_fixmatch_loss += fixmatch_loss.item()
                loss += self.params.fixmatch_lw * fixmatch_loss

            if self.params.classcontrast:
                try:
                    ux, uy, *_ = next(unlabeled_iter)
                except:
                    unlabeled_iter = iter(unlabeled_loader)
                    ux, uy, *_ = next(unlabeled_iter)
                ux0, ux1 = ux[0].cuda(), ux[1].cuda()
                with torch.no_grad():
                    prob0 = F.softmax(self.classifier(self.feature(ux0)), dim=-1)
                    mask = prob0.max(dim=-1)[0] > self.params.classcontrast_th
                logit1 = self.classifier(self.feature(ux1))
                if self.params.classcontrast_fn == 'dot':
                    prob1 = F.softmax(logit1, -1)
                    score = prob1.mm(prob0.t())
                elif self.params.classcontrast_fn == 'kl':
                    log_prob = F.log_softmax(logit1, -1)
                    log_prob = log_prob.unsqueeze(1).repeat(1, prob0.shape[0], 1)
                    prob0 = prob0.unsqueeze(0).repeat(log_prob.shape[0], 1, 1)
                    score = - F.kl_div(log_prob, prob0, reduction='none').sum(-1)
                labels = torch.arange(logit1.shape[0], dtype=torch.long).cuda()
                classcontrast_loss = F.cross_entropy(score / self.params.classcontrast_t, labels, reduction='none')
                classcontrast_loss = (classcontrast_loss * mask).mean()
                avg_classcontrast_loss += classcontrast_loss.item()
                loss += classcontrast_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.ad_align:
                if self.pseudo_align:
                    fx = fx[real_idx]
                real_loss, _ = self.discriminator_loss(fx.detach(), 1, is_feature=True)
                fake_loss, _ = self.discriminator_loss(fux.detach(), 0, is_feature=True)
                discriminator_loss = (real_loss + fake_loss) / 2
                avg_discriminator_loss += discriminator_loss.item()

                discriminator_optim.zero_grad()
                discriminator_loss.backward()
                discriminator_optim.step()

            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print_line = 'Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                               avg_loss / float(i + 1))
                if self.ad_align:
                    print_line += ' | Ad_loss {:f} | Discriminator_loss {:f}'.format(avg_ad_loss / (i + 1),
                                                                                     avg_discriminator_loss / (i + 1))
                if self.proto_align:
                    print_line += ' | Proto_loss {:f}'.format(avg_proto_loss / (i + 1))
                if self.rot_align:
                    print_line += ' | Rot_loss {:f}'.format(avg_rot_loss / (i + 1))
                if self.params.simclr:
                    print_line += ' | Simclr_loss {:f}'.format(avg_simclr_loss / (i + 1))
                if self.params.fixmatch:
                    print_line += ' | Fixmatch_loss {:f}'.format(avg_fixmatch_loss / (i + 1))
                if self.params.classcontrast:
                    print_line += ' | Classcontrast_loss {:f}'.format(avg_classcontrast_loss / (i + 1))
                print(print_line)

            if self.pseudo_align or self.proto_align:
                if self.momentum < 1:
                    self.update_teacher()

        loss_dict = {'loss': avg_loss / (i + 1)}
        if self.ad_align:
            loss_dict['ad_loss'] = avg_ad_loss / (i + 1)
            loss_dict['discriminator_loss'] = avg_discriminator_loss / (i + 1)
        if self.proto_align:
            loss_dict['proto_loss'] = avg_proto_loss / (i + 1)
        if self.rot_align:
            loss_dict['rot_loss'] = avg_rot_loss / (i + 1)
        if self.params.simclr:
            loss_dict['simclr_loss'] = avg_simclr_loss / (i + 1)
        if self.params.fixmatch:
            loss_dict['fixmatch_loss'] = avg_fixmatch_loss / (i + 1)
        if self.params.classcontrast:
            loss_dict['classcontrast_loss'] = avg_classcontrast_loss / (i + 1)
        return loss_dict

    def test_loop(self, epoch, val_loader, params):
        # if self.DBval:
        #     return self.analysis_loop(val_loader)
        # else:
        #     return -1   #no validation, just save model during iteration
        accs = []
        with torch.no_grad():
            for img, label in tqdm(val_loader):
                img, label = img.squeeze(), label.squeeze()
                n_way = params.test_n_way
                n_shot = params.n_shot
                n_query = img.shape[1] - n_shot

                support_label = torch.arange(n_way).unsqueeze(1).repeat(1, n_shot).view(-1).numpy()
                query_label = torch.arange(n_way).unsqueeze(1).repeat(1, n_query).view(-1).numpy()

                img = img.cuda()
                img = img.view(-1, *img.shape[2:])
                features = self.feature(img)
                features = F.normalize(features, dim=1)
                features = features.view(n_way, n_shot + n_query, -1)
                support_feature = features[:, :n_shot].detach().cpu().numpy().reshape(n_way * n_shot, -1)
                query_feature = features[:, n_shot:].detach().cpu().numpy().reshape(n_way * n_query, -1)

                clf = LogisticRegression(penalty='l2',
                                         random_state=0,
                                         C=1.0,
                                         solver='lbfgs',
                                         max_iter=1000,
                                         multi_class='multinomial')
                clf.fit(support_feature, support_label)
                query_pred = clf.predict(query_feature)
                acc = np.equal(query_pred, query_label).sum() / query_label.shape[0]
                accs.append(acc * 100)
        acc_mean = np.mean(accs)
        acc_std = np.std(accs)
        print('Epoch %d, Test Acc = %4.2f%% +- %4.2f%%' % (epoch, acc_mean, 1.96 * acc_std / np.sqrt(len(accs))))
        return acc_mean

    def analysis_loop(self, val_loader, record=None):
        class_file = {}
        for i, (x, y) in enumerate(val_loader):
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
        print('DB index = %4.2f' % (DB))
        return 1 / DB  # DB index: the lower the better


def DBindex(cl_data_file):
    # For the definition Davis Bouldin index (DBindex), see https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
    # DB index present the intra-class variation of the data
    # As baseline/baseline++ do not train few-shot classifier in training, this is an alternative metric to evaluate the validation set
    # Emperically, this only works for CUB dataset but not for miniImagenet dataset

    class_list = cl_data_file.keys()
    cl_num = len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append(np.mean(cl_data_file[cl], axis=0))
        stds.append(np.sqrt(np.mean(np.sum(np.square(cl_data_file[cl] - cl_means[-1]), axis=1))))

    mu_i = np.tile(np.expand_dims(np.array(cl_means), axis=0), (len(class_list), 1, 1))
    mu_j = np.transpose(mu_i, (1, 0, 2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis=2))

    for i in range(cl_num):
        DBs.append(np.max([(stds[i] + stds[j]) / mdists[i, j] for j in range(cl_num) if j != i]))
    return np.mean(DBs)

