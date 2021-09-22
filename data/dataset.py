import torch
import numpy as np
import random


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