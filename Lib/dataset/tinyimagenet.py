# To ensure fairness, we use the same code in LDAM (https://github.com/kaidic/LDAM-DRW) to produce long-tailed CIFAR datasets.

from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
import copy
from pathlib import Path
import torch

from aug.cuda import *
from aug.transforms import *
from aug.autoaug import *
from aug.randaug import *
from aug.others import *

class ImbalancedTinyImageNet(Dataset):

    def __init__(self, root, args, imb_type='exp', imb_ratio=100, train=False, transform=None):
        super().__init__()

        self.train = train
        self.args = args
        self.root = root
        self.transform_train = transform
        self.cls_num = 200
        self.data, self.targets = self.load_data()

        if self.train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, 1./imb_ratio)
            self.img_num_list = img_num_list
            self.gen_imbalanced_data(img_num_list)
        
        if 'autoaug_cifar' in args.aug_type:
            print('autoaug_cifar')
            self.aug_transform = transforms.Compose([CIFAR10Policy()])
        elif 'autoaug_svhn' in args.aug_type:
            print('autoaug_svhn')
            self.aug_transform = transforms.Compose([SVHNPolicy()])
        elif 'autoaug_imagenet' in args.aug_type:
            print('autoaug_imagenet')
            self.aug_transform = transforms.Compose([ImageNetPolicy()])
        elif 'dada_cifar' in args.aug_type:
            print('dada_cifar')
            self.aug_transform = transforms.Compose([dada_cifar()])
        elif 'dada_imagenet' in args.aug_type:
            print('dada_imagenet')
            self.aug_transform = transforms.Compose([dada_imagenet()])
        elif 'faa_cifar' in args.aug_type:
            print('faa_cifar')
            self.aug_transform = transforms.Compose([faa_cifar()])
        elif 'faa_imagenet' in args.aug_type:
            print('faa_imagenet')
            self.aug_transform = transforms.Compose([faa_imagenet()])
        elif 'randaug' in args.aug_type:
            print('randaug')
            self.aug_transform = transforms.Compose([RandAugment(2, 14)])
        elif 'none' in args.aug_type:
            self.aug_transform = transforms.Compose([])
        else:
            raise NotImplementedError
        
        max_mag = 10
        max_ops = 10
        self.min_state = 0
        self.max_state = max(max_mag, max_ops) + 1
        
        states = torch.arange(self.min_state, self.max_state)
        if self.max_state == 1:
            self.ops = torch.tensor([0])
            self.mag = torch.tensor([0])
            
        elif max_mag > max_ops:
            self.ops = (states * max_ops / max_mag).ceil().int()
            self.mag = states.int()
        else:
            self.mag = (states * max_mag / max_ops).ceil().int()
            self.ops = states.int()
        
        print(f"Magnitude set = {self.mag}")
        print(f"Operation set = {self.ops}")

        self.curr_state = torch.zeros(len(self.data))
        self.score_tmp = torch.zeros((len(self.targets), self.max_state))
        self.num_test = torch.zeros((len(self.targets), self.max_state))
        self.aug_prob = args.aug_prob
    
    def sim_aug(self, img, state, type):
        if type == 'cuda':
            return  CUDA(img, self.mag[state], self.ops[state], max_d = self.args.max_d)
        else:
            return img
        
    def load_data(self):
        data_prefix = Path(self.root)
        ann_file = data_prefix / 'wnids.txt'
        with open(f'{ann_file}') as f:
            classes = [x.strip() for x in f.readlines()]

        samples = []
        gt_labels = []
        if not self.train:
            with open(f'{data_prefix}/val/val_annotations.txt') as f:
                lines = [x.strip().split('\t') for x in f.readlines()]
                for line in lines:
                    path = f'{data_prefix}/val/images/{line[0]}'
                    img = Image.open(path)
                    if img.mode != "RGB":
                            img = img.convert('RGB')
                    img = np.asarray(img)
                    samples.append(img)

                    label = classes.index(line[1])
                    gt_labels.append(label)
        else:
            for i,c in enumerate(classes):
                with open(f'{data_prefix}/train/{c}/{c}_boxes.txt') as f:
                    image_names = [x.strip().split('\t')[0] for x in f.readlines()]
                    for image_name in image_names:
                        path = f'{data_prefix}/train/{c}/images/{image_name}'
                        img = Image.open(path)
                        if img.mode != "RGB":
                            img = img.convert('RGB')
                        img = np.asarray(img)
                        samples.append(img)
                        gt_labels.append(i)

        samples = np.array(samples)
        gt_labels = np.array(gt_labels,  dtype=np.int64)

        return samples, gt_labels
    
    def __len__(self):
        return len(self.targets)
    
    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls   

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        state = self.curr_state[index].int() if torch.rand(1) < self.aug_prob else 0
        if self.args.no_random:
            state = self.curr_state[index].int()

        if self.train:
            if len(self.transform_train) == 1:
                img = self.transform_train[0][0](img)
                img = self.aug_transform(img)
                img = CUDA(img, self.mag[state], self.ops[state])
                img = self.transform_train[0][1](img)
                return img, target, index

            elif len(self.transform_train) == 2:
                img1 = self.transform_train[0][0](img)
                img1 = self.aug_transform(img1)
                img1 = CUDA(img1, self.mag[state], self.ops[state], max_d = self.args.max_d)
                img1 = self.transform_train[0][1](img1)

                img2 = self.transform_train[1][0](img)
                img2 = self.sim_aug(img2, state, self.args.sim_type)
                img2 = self.transform_train[1][1](img2)
                
                return (img1, img2), target, index
                
            elif len(self.transform_train) == 3:
                img1 = self.transform_train[0][0](img)
                img1 = self.aug_transform(img1)
                img1 = CUDA(img1, self.mag[state], self.ops[state], max_d = self.args.max_d)
                img1 = self.transform_train[0][1](img1)

                img2 = self.transform_train[1][0](img)
                img2 = self.sim_aug(img2, state, self.args.sim_type)
                img2 = self.transform_train[1][1](img2)
                
                img3 = self.transform_train[2][0](img)
                img3 = self.sim_aug(img3, state, self.args.sim_type)
                img3 = self.transform_train[2][1](img3)
                return (img1, img2, img3), target, index

        else:
            img = self.transform_train[0][0](img)
            img = self.aug_transform(img)
            img = CUDA(img, self.mag[state], self.ops[state], rand=False , max_d = self.args.max_d)
            img = self.transform_train[0][1](img)
            return img, target, index

        return img, target, index

    def update(self):
        # Increase
        pos = torch.where((self.score_tmp == self.num_test) & (self.num_test != 0))
        self.curr_state[pos] += 1
        
        # Decrease
        pos = torch.where(self.score_tmp != self.num_test)
        self.curr_state[pos] -= 1
        
        
        self.curr_state = torch.clamp(self.curr_state, self.min_state, self.max_state-1)
        self.score_tmp *= 0
        self.num_test *= 0

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        if len(self.targets.shape)==2:
            for target in self.labels:
                annos.append({'category_id': int(target)})
        else:
            for target in self.targets:
                annos.append({'category_id': int(target)})
        return annos

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = np.array(new_targets, dtype=np.int64)

class ImbalancedTinyImageNet_val(ImbalancedTinyImageNet):

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        img = self.transform_train(img)

        return img, target, index
    

def get_tinyimagenet(root, args):
    transform_train, transform_val = get_transform(args.loss_fn, cutout = args.cutout, args=args)
    for tg in transform_train:
        if isinstance(tg[0].transforms[0], transforms.RandomCrop):
            tg[0].transforms[0] = transforms.RandomResizedCrop(32)
        if isinstance(tg[0].transforms[0], TR.RandAugment):
            tg[0].transforms.insert(0, transforms.Resize(32))
    
    transform_val.transforms.insert(0, transforms.Resize(32))
    # print(transform_train, transform_val)
    # Compose(
    # Resize(size=32, interpolation=bilinear, max_size=None, antialias=None)
    # ToTensor()
    # Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201))
    train_dataset = ImbalancedTinyImageNet(root, args, imb_ratio = args.imb_ratio, train=True, transform = transform_train)
    test_dataset = ImbalancedTinyImageNet_val(root, args, transform=transform_val)
    print (f"#Train: {len(train_dataset)}, #Test: {len(test_dataset)}")
    return train_dataset, test_dataset

