# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torchvision
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SimpleDataset, SetDataset, EpisodicBatchSampler
from abc import abstractmethod

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Resize':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass


class DatasetWithIndex:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, label, index

    def __len__(self):
        return len(self.dataset)


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file=None, data_folder=None, aug=None, proportion=1, with_idx=False): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        if data_file is not None:
            dataset = SimpleDataset(data_file, transform)
        elif not isinstance(data_folder, list):
            dataset = torchvision.datasets.ImageFolder(data_folder, transform)
        else:
            class AddLabel:
                def __init__(self, base):
                    self.base = base
                def __call__(self, label):
                    return label + self.base
            dataset = []
            n_class = 0
            for folder in data_folder:
                target_transform = AddLabel(n_class)
                dataset.append(torchvision.datasets.ImageFolder(folder, transform, target_transform))
                n_class += len(dataset[-1].classes)

            dataset = torch.utils.data.ConcatDataset(dataset)
        if proportion < 1:
            n_samples = int(len(dataset) * proportion)
            dataset = torch.utils.data.random_split(dataset, [n_samples, len(dataset)-n_samples])[0]
        if with_idx:
            dataset = DatasetWithIndex(dataset)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 12)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way, n_support, n_query, n_episode =-1):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_support = n_support
        self.n_query = n_query
        self.n_episode = n_episode

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file=None, data_folder=None,  aug=False, fix_seed=True): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        if isinstance(data_folder, list):
            dataset = SetDataset(data_file, data_folder, [self.n_support, self.n_query], transform)
        else:
            dataset = SetDataset( data_file, data_folder, self.batch_size, transform )
        if self.n_episode < 0:
            self.n_episode = int(dataset.get_sample_number() / (self.n_way * self.batch_size))
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode, fix_seed)
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 12, pin_memory = True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


