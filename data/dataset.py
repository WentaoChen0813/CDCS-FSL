# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torchvision
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
import random


identity = lambda x:x
class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, data_file, data_folder, batch_size, transform):
        self.cross_domain = False
        self.sample_number = 0
        if data_file is not None:
            with open(data_file, 'r') as f:
                self.meta = json.load(f)

            self.cl_list = np.unique(self.meta['image_labels']).tolist()

            self.sub_meta = {}
            for cl in self.cl_list:
                self.sub_meta[cl] = []

            for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
                self.sub_meta[y].append(x)
                self.sample_number += 1

            self.sub_dataloader = []
            sub_data_loader_params = dict(batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0,  # use main thread only or may receive multiple batches
                                          pin_memory=False)
            for cl in self.cl_list:
                sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform)
                self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

        elif not isinstance(data_folder, list):
            import torchvision
            dataset = torchvision.datasets.ImageFolder(data_folder)
            self.sample_number = len(dataset)
            self.sub_meta = {}
            self.cl_list = list(range(len(dataset.classes)))
            for cl in self.cl_list:
                self.sub_meta[cl] = []

            for x, y in dataset.imgs:
                self.sub_meta[y].append(x)

            self.sub_dataloader = []
            sub_data_loader_params = dict(batch_size = batch_size,
                                      shuffle = True,
                                      num_workers = 0, #use main thread only or may receive multiple batches
                                      pin_memory = False)
            for cl in self.cl_list:
                sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
                self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

        else:
            self.cross_domain = True
            import torchvision
            support_dataset = torchvision.datasets.ImageFolder(data_folder[0])
            query_dataset = torchvision.datasets.ImageFolder(data_folder[1])
            self.sample_number = len(support_dataset) + len(query_dataset)
            self.sub_support_meta = {}
            self.sub_query_meta = {}
            self.cl_list = list(range(len(support_dataset.classes)))
            for cl in self.cl_list:
                self.sub_support_meta[cl] = []
                self.sub_query_meta[cl] = []
            for x, y in support_dataset.imgs:
                self.sub_support_meta[y].append(x)
            for x, y in query_dataset.imgs:
                self.sub_query_meta[y].append(x)
            self.sub_support_dataloader = []
            self.sub_query_dataloader = []
            support_dataloader_params = dict(batch_size=batch_size[0],
                                             shuffle=True,
                                             num_workers=0,  # use main thread only or may receive multiple batches
                                             pin_memory=False)
            query_dataloader_params = dict(batch_size=batch_size[1],
                                           shuffle=True,
                                           num_workers=0,  # use main thread only or may receive multiple batches
                                           pin_memory=False)
            for cl in self.cl_list:
                sub_support_dataset = SubDataset(self.sub_support_meta[cl], cl, transform=transform)
                self.sub_support_dataloader.append(torch.utils.data.DataLoader(sub_support_dataset, **support_dataloader_params))
                sub_query_dataset = SubDataset(self.sub_query_meta[cl], cl, transform=transform)
                self.sub_query_dataloader.append(torch.utils.data.DataLoader(sub_query_dataset, **query_dataloader_params))

    def __getitem__(self,i):
        if not self.cross_domain:
            return next(iter(self.sub_dataloader[i]))
        else:
            support_img, support_label = next(iter(self.sub_support_dataloader[i]))
            query_img, query_label = next(iter(self.sub_query_dataloader[i]))
            return torch.cat([support_img, query_img]), torch.cat([support_label, query_label])

    def __len__(self):
        return len(self.cl_list)

    def get_sample_number(self):
        return self.sample_number


class EpisodeDataset:
    def __init__(self, data_folder, transform, n_way, n_support, n_query, n_episode, fix_seed=False):
        self.cross_domain = False
        self.transform = transform
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_episode = n_episode
        self.fix_seed = fix_seed

        if not isinstance(data_folder, list):
            import torchvision
            dataset = torchvision.datasets.ImageFolder(data_folder, transform=transform)
            self.dataset = dataset
            if self.n_episode < 0:
                self.n_episode = len(dataset) // (n_way*(n_support+n_query))
            self.cl_list = list(range(len(dataset.classes)))
            self.cl_idx = []
            labels = np.array(dataset.targets)
            for cl in self.cl_list:
                self.cl_idx.append(np.where(labels==cl)[0])
        else:
            self.cross_domain = True
            import torchvision
            support_dataset = torchvision.datasets.ImageFolder(data_folder[0], transform=transform)
            query_dataset = torchvision.datasets.ImageFolder(data_folder[1], transform=transform)
            self.support_dataset = support_dataset
            self.query_dataset = query_dataset
            if self.n_episode < 0:
                self.n_episode = (len(support_dataset) + len(query_dataset)) // (n_way*(n_support+n_query))
            self.cl_list = list(range(len(support_dataset.classes)))
            self.sup_cl_idx = []
            self.que_cl_idx = []
            sup_labels = np.array(support_dataset.targets)
            que_labels = np.array(query_dataset.targets)
            for cl in self.cl_list:
                self.sup_cl_idx.append(np.where(sup_labels==cl)[0])
                self.que_cl_idx.append(np.where(que_labels==cl)[0])

    def __getitem__(self,i):
        if self.fix_seed:
            random.seed(i)
        classes = random.sample(self.cl_list, k=self.n_way)
        images = []
        labels = []
        for cl in classes:
            if not self.cross_domain:
                cl_idx = list(self.cl_idx[cl])
                idx = random.sample(cl_idx, k=self.n_support+self.n_query)
                for i in idx:
                    image, label = self.dataset[i]
                    images.append(image)
                    labels.append(label)
            else:
                cl_idx = list(self.sup_cl_idx[cl])
                idx = random.sample(cl_idx, k=self.n_support)
                for i in idx:
                    image, label = self.support_dataset[i]
                    images.append(image)
                    labels.append(label)
                cl_idx = list(self.que_cl_idx[cl])
                if len(cl_idx) > self.n_query:
                    idx = random.sample(cl_idx, k=self.n_query)
                else:
                    m, n = int(self.n_query//len(cl_idx)), int(self.n_query % len(cl_idx))
                    idx = cl_idx * m
                    idx.extend(cl_idx[:n])
                for i in idx:
                    image, label = self.query_dataset[i]
                    images.append(image)
                    labels.append(label)
        images = torch.stack(images)
        images = images.view(self.n_way, self.n_support+self.n_query, *images.shape[1:])
        labels = torch.tensor(labels)
        labels = labels.view(self.n_way, -1)

        return images, labels

    def __len__(self):
        return self.n_episode


class DatasetWithoutIndex:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        img, label, _ = self.dataset[index]
        return img, label

    def __len__(self):
        return len(self.dataset)


class PseudoPairedSetDataset:
    def __init__(self, labeled_dataset, unlabeled_dataset, selected_idx, selected_pred, n_shot, n_query):
        self.n_shot = n_shot
        self.n_query = n_query
        support_dataset = DatasetWithoutIndex(unlabeled_dataset)
        query_dataset = labeled_dataset
        self.sub_query_meta = {}
        self.cl_list = np.unique(selected_pred)
        for cl in self.cl_list:
            self.sub_query_meta[cl] = []
        for x, y in query_dataset.imgs:
            if y in self.cl_list:
                self.sub_query_meta[y].append(x)
        self.sub_support_dataloader = []
        self.sub_query_dataloader = []
        support_dataloader_params = dict(batch_size=n_shot,
                                         shuffle=True,
                                         num_workers=0,  # use main thread only or may receive multiple batches
                                         pin_memory=False,
                                         drop_last=False)
        query_dataloader_params = dict(batch_size=n_query,
                                       shuffle=True,
                                       num_workers=0,  # use main thread only or may receive multiple batches
                                       pin_memory=False,
                                       drop_last=False)
        for cl in self.cl_list:
            sub_idx = selected_idx[selected_pred == cl]
            sub_support_dataset = torch.utils.data.Subset(support_dataset, sub_idx)
            self.sub_support_dataloader.append(
                torch.utils.data.DataLoader(sub_support_dataset, **support_dataloader_params))
            sub_query_dataset = SubDataset(self.sub_query_meta[cl], cl, transform=query_dataset.transform)
            self.sub_query_dataloader.append(
                torch.utils.data.DataLoader(sub_query_dataset, **query_dataloader_params))

    def __getitem__(self, i):
            support_img, support_label = next(iter(self.sub_support_dataloader[i]))
            # if n_samples is less than n_shot, then sample again with different transforms
            while len(support_label) < self.n_shot:
                extra_support_img, extra_support_label = next(iter(self.sub_support_dataloader[i]))
                n_extra = self.n_shot - len(support_label)
                support_img = torch.cat([support_img, extra_support_img[:n_extra]])
                support_label = torch.cat([support_label, extra_support_label[:n_extra]])
            query_img, query_label = next(iter(self.sub_query_dataloader[i]))
            return torch.cat([support_img, query_img]), torch.cat([support_label, query_label])

    def __len__(self):
        return len(self.cl_list)


class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
        image_path = os.path.join( self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes, fix_seed=True):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes
        self.fix_seed = fix_seed

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        seed = torch.get_rng_state()[0]
        for i in range(self.n_episodes):
            if self.fix_seed:
                torch.manual_seed(i)
            yield torch.randperm(self.n_classes)[:self.n_way]
        if self.fix_seed:
            torch.manual_seed(seed)

# class EpisodeSampler:
#     def __init__(self, label, n_batch, n_cls, n_per, fix_seed=True):
#         if n_batch > 0:
#             self.n_batch = n_batch
#         else:
#             self.n_batch = int(len(label)//(n_cls*n_per))
#         self.n_cls = n_cls
#         self.n_per = n_per
#         self.fix_seed = fix_seed
#
#         label = np.array(label)
#         self.m_ind = []
#         for i in range(max(label) + 1):
#             ind = np.argwhere(label == i).reshape(-1)
#             ind = torch.from_numpy(ind)
#             self.m_ind.append(ind)
#
#     def __len__(self):
#         return self.n_batch
#
#     def __iter__(self):
#         for i_batch in range(self.n_batch):
#             if self.fix_seed:
#                 np.random.seed(i_batch)
#             batch = []
#             # classes = torch.randperm(len(self.m_ind))[: self.n_cls]
#             classes = np.random.choice(range(len(self.m_ind)), self.n_cls, False)
#             for c in classes:
#                 l = self.m_ind[c]
#                 # pos = torch.randperm(len(l))[: self.n_per]
#                 pos = np.random.choice(range(len(l)), self.n_per, False)
#                 batch.append(l[pos])
#             batch = torch.cat(batch).reshape(-1)
#             if self.fix_seed:
#                 np.random.seed()
#             yield batch
#
# class CrossDomainEpisodeSampler:
#     def __init__(self, support_label, query_label, n_batch, n_cls, n_shot, n_query, fix_seed=True):
#         if n_batch > 0:
#             self.n_batch = n_batch
#         else:
#             self.n_batch = int(len(support_label)+len(query_label)//(n_cls*(n_shot+n_query)))
#         self.n_cls = n_cls
#         self.n_shot = n_shot
#         self.
#         self.n_per = n_per
#         self.fix_seed = fix_seed
#
#         label = np.array(label)
#         self.m_ind = []
#         for i in range(max(label) + 1):
#             ind = np.argwhere(label == i).reshape(-1)
#             ind = torch.from_numpy(ind)
#             self.m_ind.append(ind)
#
#     def __len__(self):
#         return self.n_batch
#
#     def __iter__(self):
#         for i_batch in range(self.n_batch):
#             if self.fix_seed:
#                 np.random.seed(i_batch)
#             batch = []
#             # classes = torch.randperm(len(self.m_ind))[: self.n_cls]
#             classes = np.random.choice(range(len(self.m_ind)), self.n_cls, False)
#             for c in classes:
#                 l = self.m_ind[c]
#                 # pos = torch.randperm(len(l))[: self.n_per]
#                 pos = np.random.choice(range(len(l)), self.n_per, False)
#                 batch.append(l[pos])
#             batch = torch.cat(batch).reshape(-1)
#             if self.fix_seed:
#                 np.random.seed()
#             yield batch

