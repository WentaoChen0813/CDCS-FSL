# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torch.nn as nn
import torchvision
from PIL import Image
import numpy as np
import random
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SimpleDataset, SetDataset, EpisodicBatchSampler, EpisodeDataset
from abc import abstractmethod
from data.randaugment import RandAugmentMC
import os


class TransformLoader:
    def __init__(self, image_size,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
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


# https://github.com/sthalles/SimCLR
class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class SimCLRTransform:
    def __init__(self, size, s=1):
        color_jitter = transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor(),
                                              normalize])
        self.transform = data_transforms

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2


class FixMatchTransform:
    def __init__(self, size=224, n_anchor=1, has_weak=True):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(int(size * 1.15)),
            transforms.CenterCrop(size)])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(int(size * 1.15)),
            transforms.CenterCrop(size),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.n_anchor = n_anchor
        self.has_weak = has_weak

    def __call__(self, x):
        out = []
        if self.has_weak:
            weak = self.normalize(self.weak(x))
            out.append(weak)
        for i in range(self.n_anchor):
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


class DatasetWithRotation:
    def __init__(self, dataset):
        self.dataset = dataset
        self.rot = True

    def __getitem__(self, i):
        if not self.rot:
            return self.dataset[i]
        input = list(self.dataset[i])
        img = input[0]
        rot = random.choice(range(4))
        img = torch.rot90(img, k=rot, dims=[1, 2])
        input[0] = img
        input[1] = (input[1], rot)
        return tuple(input)

    def __len__(self):
        return len(self.dataset)


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)
        self.image_size = image_size

    def get_data_loader(self, data_file=None, data_folder=None, add_label=False, aug=None, proportion=1, with_idx=False,
                        rot=False, simclr_trans=False, fixmatch_trans=False, fixmatch_anchor=1,
                        fixmatch_weak=True, drop_last=False):  # parameters that would change on train/val set
        if simclr_trans:
            transform = SimCLRTransform(self.image_size)
        elif fixmatch_trans:
            transform = FixMatchTransform(self.image_size, fixmatch_anchor, fixmatch_weak)
        else:
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
                if add_label:
                    target_transform = AddLabel(n_class)
                else:
                    target_transform = None
                dataset.append(torchvision.datasets.ImageFolder(folder, transform, target_transform))
                n_class += len(dataset[-1].classes)

            dataset = torch.utils.data.ConcatDataset(dataset)
        if proportion < 1:
            n_all = len(dataset)
            file_name = f'record/subidx_all_{n_all}_proportion_{proportion:.1f}.npy'
            if os.path.exists(file_name):
                sub_idx = np.load(file_name)
            else:
                n_samples = int(n_all * proportion)
                sub_idx = np.random.choice(n_all, n_samples, replace=False)
                np.save(file_name, sub_idx)
            dataset = torch.utils.data.Subset(dataset, sub_idx)
        if with_idx:
            dataset = DatasetWithIndex(dataset)
        if rot:
            dataset = DatasetWithRotation(dataset)
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=12, drop_last=drop_last)
        if simclr_trans:
            data_loader_params['drop_last'] = True
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader


class SetDataManager(DataManager):
    def __init__(self, image_size, n_way, n_support, n_query, n_episode=-1):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_support = n_support
        self.n_query = n_query
        self.n_episode = n_episode

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file=None, data_folder=None, aug=False,
                        fix_seed=False):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        # if isinstance(data_folder, list):
        #     dataset = SetDataset(data_file, data_folder, [self.n_support, self.n_query], transform)
        # else:
        #     dataset = SetDataset( data_file, data_folder, self.batch_size, transform )
        dataset = EpisodeDataset(data_folder, transform, self.n_way, self.n_support, self.n_query, self.n_episode,
                                 fix_seed)
        # if self.n_episode < 0:
        #     self.n_episode = int(dataset.get_sample_number() / (self.n_way * self.batch_size))
        # sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode, fix_seed)
        data_loader_params = dict(num_workers=12, pin_memory=True, batch_size=1)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


