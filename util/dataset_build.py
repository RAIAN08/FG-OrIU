import numpy as np
import os, argparse, sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from image_iter import CustomSubset
import random

import sys
sys.path.append('.')



def create_subset_dataset(split1_dataset, data_ratio):

    class_samples = {}
    for sample, label in split1_dataset.samples:
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append((sample, label))

    subset_samples = []
    for label, samples in class_samples.items():
        num_samples = len(samples)
        num_selected = int(num_samples * data_ratio)
        selected_samples = random.sample(samples, num_selected) 
        subset_samples.extend(selected_samples)


    subset_samples_paths, subset_labels = zip(*subset_samples)

    subset_dataset = datasets.ImageFolder(root=split1_dataset.root, transform=split1_dataset.transform)
    subset_dataset.samples = [(path, label) for path, label in zip(subset_samples_paths, subset_labels)]
    subset_dataset.targets = list(subset_labels)
    
    subset_dataset.classes = split1_dataset.classes
    subset_dataset.class_to_idx = split1_dataset.class_to_idx

    return subset_dataset

def Ratio_dataset(dataset, data_ratio):
    len_dataset = len(dataset)

    subset_size = int(len_dataset * data_ratio)
    subset_indices = torch.randperm(len_dataset)[
        :subset_size
    ]
    dataset_sub = CustomSubset(
        dataset, subset_indices
    )
    return dataset_sub

def Fetch_from_Original_Dataset(
    original_dataset, 
    class_order_list, 
    # each_task_num, 
    random_ratio, 
    start, 
    end,
    dataset_type,
    data_ratio=None,
    transform=ToTensor(),
    ):

    num_classes = len(original_dataset.classes)

    split1_class_indices = class_order_list[
        start:end
    ]  # Does not include split1_end

    # create a dataset for interval 1
    split1_samples = [
        (sample, label)
        for sample, label in original_dataset.samples
        if label in split1_class_indices
    ]
    split1_dataset = ImageFolder(root=original_dataset.root, transform=transform)
    split1_dataset.samples = split1_samples
    split1_dataset.targets = [label for _, label in split1_samples]
    split1_dataset.classes = [original_dataset.classes[idx] for idx in split1_class_indices]
    split1_dataset.class_to_idx = {
        class_name: i for i, class_name in enumerate(split1_dataset.classes)
    }
    
    if data_ratio is not None:
        split1_dataset = create_subset_dataset(split1_dataset, data_ratio)

    if random_ratio is not None:

        assert dataset_type == 'dr', "random_ratio un consistent with dataset_type"

        if random_ratio == 0:
            return split1_dataset, None, split1_dataset
        elif random_ratio == 1:
            return split1_dataset, split1_dataset, None
        else:
            num_classes_to_select = int(len(split1_dataset.classes) * random_ratio)
            

            selected_classes = random.sample(split1_dataset.classes, num_classes_to_select)
            

            selected_class_indices = {split1_dataset.class_to_idx[class_name] for class_name in selected_classes}
            remaining_class_indices = set(split1_dataset.class_to_idx.values()) - selected_class_indices


            selected_samples = [(sample, label) for sample, label in split1_dataset.samples if label in selected_class_indices]
            remaining_samples = [(sample, label) for sample, label in split1_dataset.samples if label in remaining_class_indices]


            access_subset = datasets.ImageFolder(root=split1_dataset.root, transform=split1_dataset.transform)
            access_subset.samples = selected_samples
            access_subset.targets = [label for _, label in selected_samples]
            access_subset.classes = [split1_dataset.classes[idx] for idx in selected_class_indices]
            access_subset.class_to_idx = {class_name: i for i, class_name in enumerate(access_subset.classes)}

            unaccess_subset = datasets.ImageFolder(root=split1_dataset.root, transform=split1_dataset.transform)
            unaccess_subset.samples = remaining_samples
            unaccess_subset.targets = [label for _, label in remaining_samples]
            unaccess_subset.classes = [split1_dataset.classes[idx] for idx in remaining_class_indices]
            unaccess_subset.class_to_idx = {class_name: i for i, class_name in enumerate(unaccess_subset.classes)}


            return split1_dataset, access_subset, unaccess_subset

    return split1_dataset, None, None


def Fetch_all(args, dataset, task_id, class_order_list, transform, NUM_CLASS, train=True, data_ratio=0.05):
    print(len(dataset.classes))
    # train
    if train == True:
        data_ratio = data_ratio
    else:
        data_ratio = 1

    start_forget = 0 + args.per_forget_cls * task_id
    end_forget = 0 + args.per_forget_cls * (task_id + 1)
    print(start_forget, end_forget)

    forget_dataset_train, _, _ = Fetch_from_Original_Dataset(
        original_dataset=dataset,
        class_order_list=class_order_list,
        random_ratio=None,
        start=start_forget,
        end=end_forget,
        dataset_type='df',
        transform=transform,
        data_ratio=data_ratio
        )
    
    start_remain = end_forget
    end_remain = NUM_CLASS
    print(start_remain, end_remain)

    # remain_dataset_train, access_remain_dataset_train, unaccess_remain_dataset_train = Fetch_from_Original_Dataset(
    access_remain_dataset_train, remain_dataset_train, unaccess_remain_dataset_train = Fetch_from_Original_Dataset(
        original_dataset=dataset,
        class_order_list=class_order_list,
        random_ratio=args.random_ratio,
        start=start_remain,
        end=end_remain,
        dataset_type='dr',
        transform=transform,
        data_ratio=data_ratio
        )

    if task_id == 0:
        old_start = -999
        old_end = -999
        old_dataset_test = None
    else:
        old_start = 0
        old_end = 0 + args.per_forget_cls * task_id
        old_dataset_test = Fetch_from_Original_Dataset(
            original_dataset=dataset,
            class_order_list=class_order_list,
            random_ratio=None,
            start=old_start,
            end=old_end,
            dataset_type='do',
            transform=transform,
            )
    print('start_forget', start_forget, 'end_forget', end_forget, 'start_remain', start_remain)
    print('end_remain', end_remain, 'old_start', old_start, 'old_end', old_end)
    
    # if args.random_ratio is not None:
    return forget_dataset_train, remain_dataset_train, access_remain_dataset_train, unaccess_remain_dataset_train, old_dataset_test
    # else:
    #     return forget_dataset_train, remain_dataset_train, old_dataset_test

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)

def Dataloader_build(dataset, batch_size, shuffle, num_workers, SEED, drop_last=False):
    sub_generator = torch.Generator()
    sub_generator.manual_seed(SEED)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=sub_generator,
    )
    return loader

from engine_cl import train_one_epoch, eval_data
def Cal_acc(record_table, task_id, loader1, loader2, loader3, loader4, loader5, BACKBONE, DEVICE, mode=''):
    idx=0

    # table_1_before_train[task_i] = {'forget': [], 'access_remain': [], 'old': [], 'all_remain': [], 'unaccess_remain': []}
    # table_1_after_train[task_i] =  {'forget': [], 'access_remain': [], 'old': [], 'all_remain': [], 'unaccess_remain': []}

    name_list = ['forget', 'access_remain', 'old', 'all_remain', 'unaccess_remain']
    for item in [loader1, loader2, loader3, loader4, loader5]:
        if item is not None:
            acc_record = eval_data(
                model=BACKBONE,
                dataloader=item,
                device=DEVICE,
                mode=mode,
            )
        else:
            acc_record=-999
        name_of_loader = name_list[idx]
        record_table[task_id][name_of_loader] = acc_record
        idx += 1
    
    return record_table
