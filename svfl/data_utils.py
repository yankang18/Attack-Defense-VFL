import os
import ssl

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

import pandas as pd
from dataset.bhi_dataset import BHIDataset2Party
from dataset.ctr_dataset import Avazu2party, Criteo2party
from dataset.nuswide_dataset import NUSWIDEDataset
from dataset.simple_dataset import SimpleDataset, SimpleTwoPartyDataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler


ssl._create_default_https_context = ssl._create_unverified_context

tp = transforms.ToTensor()
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
     ])

transform_fn = transforms.Compose([
    transforms.ToTensor()
])


def shuffle_data(data):
    len = data.shape[0]
    perm_idxs = np.random.permutation(len)
    return data[perm_idxs]


def load_default_credit_data(ds_file_name, shuffle=False):
    dataframe = pd.read_csv(ds_file_name, skipinitialspace=True)
    samples = dataframe.values
    if shuffle:
        samples = shuffle_data(samples)
    return samples


def load_vehicle_data(ds_file_name, num_classes=2, shuffle=False):
    dataframe = pd.read_csv(ds_file_name, skipinitialspace=True)
    print("label 0: ", dataframe[dataframe['0'] == 0].shape)
    print("label 1: ", dataframe[dataframe['0'] == 1].shape)
    print("label 2: ", dataframe[dataframe['0'] == 2].shape)
    if num_classes == 2:
        dataframe = dataframe[(dataframe['0'] == 0) | (dataframe['0'] == 1)]
        print("[INFO] select samples with label 0 or 1.")
    samples = dataframe.values
    if shuffle:
        samples = shuffle_data(samples)
    return samples


def get_default_credit_data_A(ds_file_name, shuffle=False):
    samples = load_default_credit_data(ds_file_name, shuffle)
    # return samples
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(samples[:, 1:])
    return scaled


def get_default_credit_data_B(ds_file_name, shuffle=False):
    samples = load_default_credit_data(ds_file_name, shuffle)
    # return samples[:, 2:], samples[:, 1]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(samples[:, 2:])
    return scaled, samples[:, 1]


def get_default_credit_data(data_dir, shuffle=False):
    data_a = get_default_credit_data_A(os.path.join(data_dir, "default_credit/default_credit_a.csv"), shuffle=shuffle)
    data_b, label = get_default_credit_data_B(os.path.join(data_dir, "default_credit/default_credit_b.csv"), shuffle=shuffle)
    return data_a, data_b, label


def get_vehicle_data(data_dir, num_classes=2, shuffle=False):
    samples = load_vehicle_data(os.path.join(data_dir, "vehicle/vehicle_train.csv"), num_classes, shuffle=shuffle)

    # the first column in the vehicle_train.csv is the label, the 2nd to 51th columns belong
    # to the first client, while the 52th to 101th columns belong to the second client.
    return samples[:, 1:51], samples[:, 51:101], samples[:, 0]


def label_to_one_hot(target, num_classes=10):
    target = target.cpu()
    target = torch.unsqueeze(target, 1)
    one_hot_target = torch.zeros(target.size(0), num_classes)
    one_hot_target.scatter_(1, target, 1)
    return one_hot_target


def fetch_parties_data(dataset_name, data, device="cpu"):
    """
    Fetch data for each party from data according to the specific dataset.

    :param dataset_name: the name of the dataset.
    :param data: the data loaded from the dataset.
    :param device: the device for running the code.
    :return: data for each party.
    """

    if dataset_name in ['nuswide2', 'nuswide10', 'bhi', 'ctr_avazu', 'criteo']:
        # data, label = samples
        data_a = data[0]
        data_b = data[1]
    elif dataset_name == 'cifar10' or dataset_name == 'cifar100' or dataset_name == 'cifar2':
        # data, label = samples
        data_a = data[:, :, :16, :]
        data_b = data[:, :, 16:, :]
    elif dataset_name == 'mnist' or dataset_name == 'fmnist':
        # data, label = samples
        data_a = data[:, :, :14, :]
        data_b = data[:, :, 14:, :]
    elif dataset_name == "default_credit" or dataset_name == "vehicle":
        (data_a, data_b) = data
    else:
        raise Exception("Does not support dataset:{}".format(dataset_name))
    return data_a.to(device), data_b.to(device)


def get_data_loader(dst, batch_size, num_workers=0, shuffle=False):
    # return torch.utils.data.DataLoader(dst, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    return DataLoader(dst, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_print_interval(dataset_name):
    print_interval = 10
    if dataset_name == 'cifar10':
        print_interval = 1
    elif dataset_name == 'mnist':
        print_interval = 10
    elif dataset_name == 'nuswide':
        print_interval = 10
    return print_interval


def get_class_i(dataset, label_set):
    """
    Select data samples according to labels specified in the label set.

    :param dataset: the dataset containing samples
    :param label_set: the set of labels, the samples of which to be selected
    :return:
    """
    gt_data = []
    gt_labels = []
    # num_cls = len(label_set)
    for j in range(len(dataset)):
        img, label = dataset[j]
        if label in label_set:
            label_new = label_set.index(label)
            gt_data.append(img if torch.is_tensor(img) else tp(img))
            gt_labels.append(label_new)

    gt_data = torch.stack(gt_data)
    return gt_data, gt_labels


def fetch_classes(num_classes):
    return np.arange(num_classes).tolist()


def fetch_image_data_and_label(dataset, num_classes):
    class_labels = fetch_classes(num_classes)
    return get_class_i(dataset, class_labels)


def get_dataset(dataset_name, **args):
    """
    Get datasets for the specified dataset.

    :param dataset_name: the name of the dataset.
    :param args: various arguments according to different scenaros.
    :return: training dataset, testing dataset, input dimension for each party, and the number of classes
    """

    data_dir = args["data_dir"]
    col_names = None
    pos_ratio = 0.5

    if dataset_name == "cifar10":
        half_image = 16
        input_dims = [half_image * half_image * 2, half_image * half_image * 2]
        num_classes = 10
        train_dst = datasets.CIFAR10(os.path.join(data_dir, "cifar10"), download=False, train=True, transform=transform)
        test_dst = datasets.CIFAR10(os.path.join(data_dir, "cifar10"), download=False, train=False, transform=transform)
    elif dataset_name == "cifar2":
        half_image = 16
        input_dims = [half_image * half_image * 2, half_image * half_image * 2]
        num_classes = 2
        train_dst = datasets.CIFAR10(os.path.join(data_dir, "cifar10"), download=False, train=True, transform=transform)
        data, label = fetch_image_data_and_label(train_dst, num_classes)
        train_dst = SimpleDataset(data, label)
        test_dst = datasets.CIFAR10(os.path.join(data_dir, "cifar10"), download=False, train=False, transform=transform)
        data, label = fetch_image_data_and_label(test_dst, num_classes)
        test_dst = SimpleDataset(data, label)
    elif dataset_name == "mnist":
        half_image = 14
        input_dims = [half_image * half_image * 2, half_image * half_image * 2]
        num_classes = 10
        train_dst = datasets.MNIST(os.path.join(data_dir, "cifar10"), download=False, train=True, transform=transform_fn)
        test_dst = datasets.MNIST(os.path.join(data_dir, "cifar10"), download=False, train=False, transform=transform_fn)
    elif dataset_name == "fmnist":
        half_image = 14
        input_dims = [half_image * half_image * 2, half_image * half_image * 2]
        num_classes = 10
        train_dst = datasets.FashionMNIST(os.path.join(data_dir, "fmnist"), download=True, train=True, transform=transform_fn)
        test_dst = datasets.FashionMNIST(os.path.join(data_dir, "fmnist"), download=True, train=False, transform=transform_fn)
    elif dataset_name == 'nuswide2':
        # input_dims = [634, 1000]
        input_dims = [1000, 634]  # passive party has text data, while active party has image data
        num_classes = 2
        is_imbal = args.get("imbal")
        if is_imbal:
            sel_lbls = ['clouds', 'lake']  # imbalance labels, pos:neg -> 1:9
            pos_ratio = 0.1
        else:
            sel_lbls = ['water', 'grass']  # balanced labels, pos:neg -> 1:2
            pos_ratio = 0.33
        print("[INFO] NUSWIDE-2 select labels:{}".format(sel_lbls))
        train_dst = NUSWIDEDataset(data_dir, sel_lbls, 'train')
        test_dst = NUSWIDEDataset(data_dir, sel_lbls, 'test')
    elif dataset_name == 'nuswide10':
        # input_dims = [634, 1000]
        input_dims = [1000, 634]  # passive party has text data, while active party has image data
        num_classes = 10
        sel_lbls = ['sky', 'clouds', 'person', 'water', 'animal', 'grass', 'buildings', 'window', 'plants', 'lake']
        print("[INFO] NUSWIDE-10 select labels:{}".format(sel_lbls))
        train_dst = NUSWIDEDataset(data_dir, sel_lbls, 'train')
        test_dst = NUSWIDEDataset(data_dir, sel_lbls, 'test')
    elif dataset_name == "vehicle":
        num_classes = 2  # pos:neg -> 21:18
        data_a, data_b, label = get_vehicle_data(data_dir, num_classes=num_classes, shuffle=True)
        print("[INFO] vehicle data_a:{}, data_b:{}, label:{}".format(data_a.shape, data_b.shape, label.shape))
        print("[INFO] number of positive samples {}".format(np.sum(label)))
        input_dims = [data_a.shape[1], data_b.shape[1]]
        num_samples = data_a.shape[0]
        num_train = int(num_samples * 0.8)
        train_dst = SimpleTwoPartyDataset(data_a[:num_train], data_b[:num_train], label[:num_train])
        test_dst = SimpleTwoPartyDataset(data_a[num_train:], data_b[num_train:], label[num_train:])
    elif dataset_name == "bhi":
        num_classes = 2  # pos:neg -> 1:2.5
        pos_ratio = 0.3
        train_dst = BHIDataset2Party(os.path.join(data_dir, "bhi"), "train", 50, 50, 2)
        test_dst = BHIDataset2Party(os.path.join(data_dir, "bhi"), "test", 50, 50, 2)
        input_dims = [50, 50]
    elif dataset_name == "default_credit":
        num_classes = 2
        data_a, data_b, label = get_default_credit_data(data_dir, shuffle=True)
        print("[INFO] default_credit data_a:{}, data_b:{}, label:{}".format(data_a.shape, data_b.shape, label.shape))
        print("[INFO] number of positive samples {}".format(np.sum(label)))
        input_dims = [data_a.shape[1], data_b.shape[1]]
        num_samples = data_a.shape[0]
        num_train = int(num_samples * 0.8)
        train_dst = SimpleTwoPartyDataset(data_a[:num_train], data_b[:num_train], label[:num_train])
        test_dst = SimpleTwoPartyDataset(data_a[num_train:], data_b[num_train:], label[num_train:])
    elif dataset_name == "ctr_avazu":
        input_dims = [11, 10]
        num_classes = 2
        sub_data_dir = "avazu"
        data_dir = os.path.join(data_dir, sub_data_dir)
        train_dst = Avazu2party(data_dir, 'Train', 2, 32)
        test_dst = Avazu2party(data_dir, 'Test', 2, 32)
        col_names = train_dst.feature_list
        # print("[INFO] avazu col names:{}".format(col_names))
    elif dataset_name == "criteo":
        input_dims = [11, 10]
        num_classes = 2
        pos_ratio = 0.1
        sub_data_dir = "criteo"
        data_dir = os.path.join(data_dir, sub_data_dir)
        train_dst = Criteo2party(data_dir, 'Train', 2, 32)
        test_dst = Criteo2party(data_dir, 'Test', 2, 32)
        col_names = train_dst.feature_list
        # print("[INFO] criteo col names:{}".format(col_names))
    else:
        raise Exception("Does not support dataset [{}] for now.".format(dataset_name))

    return train_dst, test_dst, test_dst, input_dims, num_classes, pos_ratio, col_names


def get_data_dict(dataset_name, **args):
    batch_size = args["batch_size"]
    train_dst, val_dst, test_dst, input_dims, num_classes, pos_ratio, col_names = get_dataset(dataset_name, **args)

    num_workers = args.get("num_workers")
    num_workers = 0 if num_workers is None else num_workers

    train_loader = get_data_loader(train_dst, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = get_data_loader(val_dst, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)
    test_loader = get_data_loader(test_dst, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)
    data_loader_dict = {"train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader}

    data_dict = dict()
    data_dict['dataset_name'] = dataset_name
    data_dict['num_classes'] = num_classes
    data_dict['input_dims'] = input_dims
    data_dict['data_loader_dict'] = data_loader_dict
    data_dict['col_names'] = col_names

    return data_dict


def fetch_batch_data(dataset_name, dataset, batch_size, device="cpu"):
    if dataset_name in ['nuswide2', 'nuswide10', 'bhi', 'ctr_avazu', 'criteo']:
        batch_data_a, batch_data_b, batch_label = [], [], []
        num_samples = len(dataset)
        # print("num_samples:", num_samples)
        for i in range(0, batch_size):
            sample_idx = torch.randint(num_samples, size=(1,)).item()
            data, label = dataset[sample_idx]
            batch_data_a.append(data[0])
            batch_data_b.append(data[1])
            batch_label.append(label)

        batch_data_a = torch.stack(batch_data_a).to(device)
        batch_data_b = torch.stack(batch_data_b).to(device)
        batch_label = torch.stack(batch_label).to(device)

        return batch_data_a, batch_data_b, batch_label

    elif dataset_name in ["mnist", "fmnist", "cifar2", "cifar10", "cifar100"]:
        batch_data, batch_label = [], []
        # randomly select batch_size amount of samples
        for i in range(0, batch_size):
            sample_idx = torch.randint(len(dataset), size=(1,)).item()
            data, label = dataset[sample_idx]
            batch_data.append(data)
            batch_label.append(label)

        if dataset_name == "mnist" or dataset_name == "fmnist":
            image_half_dim = 14
            batch_data = torch.stack(batch_data)
            batch_data_a = torch.reshape(batch_data[:, :, :image_half_dim, :], (-1, 392)).to(device)
            batch_data_b = torch.reshape(batch_data[:, :, image_half_dim:, :], (-1, 392)).to(device)
            batch_label = torch.stack(batch_label).to(device)
        else:
            # for cifar2, cifar10 and cifar100
            image_half_dim = 16
            batch_data = torch.stack(batch_data)
            batch_data_a = batch_data[:, :, :image_half_dim, :]
            batch_data_b = batch_data[:, :, image_half_dim:, :]
            batch_label = torch.stack(batch_label).to(device)
        return batch_data_a, batch_data_b, batch_label
    elif dataset_name == "default_credit" or dataset_name == "vehicle":
        batch_data_a, batch_data_b, batch_label = [], [], []
        for i in range(0, batch_size):
            sample_idx = torch.randint(len(dataset), size=(1,)).item()
            (data_a, data_b), label = dataset[sample_idx]
            batch_data_a.append(data_a)
            batch_data_b.append(data_b)
            batch_label.append(label)
        batch_data_a = torch.stack(batch_data_a).to(device)
        batch_data_b = torch.stack(batch_data_b).to(device)
        batch_label = torch.stack(batch_label).to(device)
        return batch_data_a, batch_data_b, batch_label
