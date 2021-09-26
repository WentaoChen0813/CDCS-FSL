# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torchvision
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import EpisodeDataset
from abc import abstractmethod
from data.randaugment import RandAugmentMC


class TransformLoader:
    def __init__(self, image_size, keep_ratio=False,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.keep_ratio = keep_ratio
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == 'ImageJitter':
            method = add_transforms.ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomSizedCrop':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Resize':
            # Attention: StrongAugment resize smaller edge to target size
            if self.keep_ratio:
                return method(int(self.image_size * 1.15))
            else:
                return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(self, aug=False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


class StrongAugment:
    def __init__(self, size=224, n_weak=0, n_strong=2):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.5, 1))])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # Attention: resize smaller edge to target size
            transforms.Resize(int(size * 1.15)),
            transforms.CenterCrop(size),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.n_strong = n_strong
        self.n_weak = n_weak

    def __call__(self, x):
        out = []
        for i in range(self.n_weak):
            weak = self.normalize(self.weak(x))
            out.append(weak)
        for i in range(self.n_strong):
            strong = self.normalize(self.strong(x))
            out.append(strong)
        if len(out) == 1:
            return out[0]
        return out


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
        self.image_size = image_size

    def get_data_loader(self, data_folder=None, with_idx=False, aug=None, aug_type='none', drop_last=False):  # parameters that would change on train/val set
        if aug_type == 'strong':
            transform = StrongAugment(self.image_size, n_weak=0, n_strong=2)
        elif aug_type == 'weak':
            transform = StrongAugment(self.image_size, n_weak=2, n_strong=0)
        else:
            transform = self.trans_loader.get_composed_transform(aug)

        if not isinstance(data_folder, list):
            dataset = torchvision.datasets.ImageFolder(data_folder, transform)
        else:
            dataset = []
            for folder in data_folder:
                dataset.append(torchvision.datasets.ImageFolder(folder, transform))
            dataset = torch.utils.data.ConcatDataset(dataset)
        if with_idx:
            dataset = DatasetWithIndex(dataset)
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=12, drop_last=drop_last)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader


class SetDataManager(DataManager):
    def __init__(self, image_size, n_way, n_support, n_query, n_episode=-1, aug_type='none'):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_support = n_support
        self.n_query = n_query
        self.n_episode = n_episode
        if aug_type == 'none':
            self.trans_loader = TransformLoader(image_size, keep_ratio=False)
        else:
            self.trans_loader = TransformLoader(image_size, keep_ratio=True)


    def get_data_loader(self, data_file=None, data_folder=None, aug=False,
                        fix_seed=False):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = EpisodeDataset(data_folder, transform, self.n_way, self.n_support, self.n_query, self.n_episode,
                                 fix_seed)
        data_loader_params = dict(num_workers=12, pin_memory=True, batch_size=1)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader