import os
import os.path as osp
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np

np.random.seed(2019)
import random

random.seed(2019)
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
 

class Data:
    def __init__(self, args):
        pin_memory = False
        if args.GPUS is not None:
            pin_memory = True

        traindir = os.path.join(args.DATASET.data_path, 'train')
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        trainset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2),
                transforms.ToTensor(),
                normalize,
            ]))

        self.trainLoader = DataLoader(
            trainset,
            batch_size=args.DATASET.train_batch_size,
            shuffle=True,
            num_workers=args.DATASET.workers,
            pin_memory=pin_memory)

        testset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
            
        val_idx_filename = "../data/imagenet_train_val_split_idx.pickle"
        # print("len(train_dataset)", len(trainset))
        if not osp.exists(val_idx_filename):
            val_size = 10000
            val_idx = []
            cls_start, cls_end = 0, 0
            for c_id in range(1000):
                for i in range(cls_start, len(trainset)):
                    if trainset[i][1] == c_id:
                        cls_end = i + 1
                    else:
                        break
                c_list = list(range(cls_start, cls_end))
                print("cid:{}, c_start:{}, c_end:{}".format(c_id, cls_start, cls_end))
                print(int(val_size / 1000))
                c_sample = random.sample(c_list, int(val_size / 1000))
                val_idx += c_sample
                cls_start = cls_end
            print("len of val_size:{}".format(len(val_idx)))
            pickle.dump(val_idx, open(val_idx_filename, "wb"))
        else:
            val_idx = pickle.load(open(val_idx_filename, "rb"))
        val_sampler = SubsetRandomSampler(val_idx)

        self.validLoader = DataLoader(
            testset,
            batch_size=args.DATASET.eval_batch_size,
            shuffle=False,
            num_workers=args.DATASET.workers,
            pin_memory=True,
            sampler=val_sampler)
        # print("len(val_dataset)", len(trainset))



