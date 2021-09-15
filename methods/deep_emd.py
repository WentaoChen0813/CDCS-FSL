# This code is modified from https://github.com/icoz69/DeepEMD

import backbone
import torch
import torch.nn as nn
import cv2
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

class DeepEMD(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(DeepEMD, self).__init__( model_func,  n_way, n_support, model_args={'flatten': False})
        self.loss_fn = nn.CrossEntropyLoss()
        self.temperature = 12.5
        self.H = 7
        self.W = 7
        self.sfc_lr = 0.1
        self.sfc_update_step = 100
        self.sfc_bs = 4

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = y_query.cuda()

        scores = self.set_forward(x)

        return self.loss_fn(scores, y_query )

    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous().view(self.n_way, self.n_support, -1, self.H, self.W)
        if self.n_support == 1:
            z_proto = z_support.squeeze()
        else:
            z_proto = self.get_sfc(z_support)
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1, self.H, self.W)

        scores = self.emd_forward_1shot(z_proto, z_query)
        return scores

    def get_sfc(self, support):
        # init the proto
        SFC = support.mean(dim=1).clone().detach()
        SFC = nn.Parameter(SFC.detach(), requires_grad=True)

        optimizer = torch.optim.SGD([SFC], lr=self.sfc_lr, momentum=0.9, dampening=0.9, weight_decay=0)

        support = support.view(-1, *support.shape[2:])
        # crate label for finetune
        label_shot = torch.arange(self.n_way).unsqueeze(-1).repeat(1, self.n_support).view(-1)
        label_shot = label_shot.type(torch.cuda.LongTensor)

        with torch.enable_grad():
            for k in range(0, self.sfc_update_step):
                rand_id = torch.randperm(self.n_way * self.n_support).cuda()
                for j in range(0, self.n_way * self.n_support, self.sfc_bs):
                    selected_id = rand_id[j: min(j + self.sfc_bs, self.n_way * self.n_support)]
                    batch_shot = support[selected_id, :]
                    batch_label = label_shot[selected_id]
                    optimizer.zero_grad()
                    logits = self.emd_forward_1shot(SFC, batch_shot.detach())
                    loss = F.cross_entropy(logits, batch_label)
                    loss.backward()
                    optimizer.step()
        return SFC

    def emd_forward_1shot(self, proto, query):
        weight_1 = self.get_weight_vector(query, proto)
        weight_2 = self.get_weight_vector(proto, query)

        proto = self.normalize_feature(proto)
        query = self.normalize_feature(query)

        similarity_map = self.get_similiarity_map(proto, query)  # Nq, way, HW, HW

        logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='opencv')
        return logits

    def get_weight_vector(self, A, B):

        M = A.shape[0]
        N = B.shape[0]

        B = F.adaptive_avg_pool2d(B, [1, 1])
        B = B.repeat(1, 1, A.shape[2], A.shape[3])

        A = A.unsqueeze(1)
        B = B.unsqueeze(0)

        A = A.repeat(1, N, 1, 1, 1)
        B = B.repeat(M, 1, 1, 1, 1)

        combination = (A * B).sum(2)
        combination = combination.view(M, N, -1)
        combination = F.relu(combination) + 1e-3
        return combination

    def normalize_feature(self, x):
        x = x - x.mean(1).unsqueeze(1)
        return x

    def get_similiarity_map(self, proto, query):
        way = proto.shape[0]
        num_query = query.shape[0]
        query = query.view(query.shape[0], query.shape[1], -1)
        proto = proto.view(proto.shape[0], proto.shape[1], -1)

        proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
        query = query.unsqueeze(1).repeat([1, way, 1, 1])
        proto = proto.permute(0, 1, 3, 2)
        query = query.permute(0, 1, 3, 2)
        feature_size = proto.shape[-2]

        proto = proto.unsqueeze(-3)
        query = query.unsqueeze(-2)
        query = query.repeat(1, 1, 1, feature_size, 1)
        similarity_map = F.cosine_similarity(proto, query, dim=-1)

        return similarity_map

    def get_emd_distance(self, similarity_map, weight_1, weight_2, solver='opencv'):
        num_query = similarity_map.shape[0]
        num_proto = similarity_map.shape[1]
        num_node=weight_1.shape[-1]

        if solver == 'opencv':  # use openCV solver
            for i in range(num_query):
                for j in range(num_proto):
                    _, flow = emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])

                    similarity_map[i, j, :, :] =(similarity_map[i, j, :, :])*torch.from_numpy(flow).cuda()

            temperature=(self.temperature/num_node)
            logitis = similarity_map.sum(-1).sum(-1) *  temperature
            return logitis
        else:
            raise ValueError('Unknown Solver')

def emd_inference_opencv(cost_matrix, weight1, weight2):
    # cost matrix is a tensor of shape [N,N]
    cost_matrix = cost_matrix.detach().cpu().numpy()

    weight1 = F.relu(weight1) + 1e-5
    weight2 = F.relu(weight2) + 1e-5

    weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
    weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()

    cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
    return cost, flow