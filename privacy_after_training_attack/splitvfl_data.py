import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets
from dataset.simple_dataset import SimpleDataset, SimpleTwoPartyDataset

tp = transforms.ToTensor()
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
     ])

transform_fn = transforms.Compose([
    transforms.ToTensor()
])


def select_samples_by_labels(dataset, label_set, label_sample_dict):
    """
    Select data samples according to labels specified in the label set.

    :param dataset: the dataset containing samples
    :param label_set: the set of labels, the samples of which to be selected
    :param label_sample_dict:
    :return:
    """

    num_recorder = {str(lbl): 0 for lbl in label_set}
    gt_data = []
    gt_labels = []
    # num_cls = len(label_set)
    for j in range(len(dataset)):
        data, label = dataset[j]
        if isinstance(label, torch.Tensor):
            label = label.item()
        if label in label_set:
            if num_recorder[str(label)] < label_sample_dict[str(label)]:
                num_recorder[str(label)] = num_recorder[str(label)] + 1
                label_new = label_set.index(label)
                # gt_data.append(data if torch.is_tensor(data) else tp(data))
                gt_data.append(data)
                gt_labels.append(label_new)
            # gt_labels.append(label_to_onehot(torch.Tensor([label_new]).long(),num_classes=num_cls))
    # gt_labels = label_to_onehot(torch.Tensor(gt_labels).long(), num_classes=num_cls)
    # gt_data = np.stack(gt_data)
    return SimpleDataset(gt_data, gt_labels)


def select_samples_by_labels_from_2party_dataset(two_party_dataset, label_set, label_sample_dict):
    """
    Select data samples according to labels specified in the label set.

    :param two_party_dataset: the dataset containing samples
    :param label_set: the set of labels, the samples of which to be selected
    :param label_sample_dict:
    :return:
    """

    num_recorder = {str(lbl): 0 for lbl in label_set}
    gt_data_a = []
    gt_data_b = []
    gt_labels = []
    for j in range(len(two_party_dataset)):
        (data_a, data_b), label = two_party_dataset[j]
        if isinstance(label, torch.Tensor):
            label = label.cpu().item()
        if label in label_set:
            if num_recorder[str(label)] < label_sample_dict[str(label)]:
                num_recorder[str(label)] = num_recorder[str(label)] + 1
                label_new = label_set.index(label)
                # gt_data.append(data if torch.is_tensor(data) else tp(data))
                gt_data_a.append(data_a)
                gt_data_b.append(data_b)
                gt_labels.append(label_new)
            # gt_labels.append(label_to_onehot(torch.Tensor([label_new]).long(),num_classes=num_cls))
    # gt_labels = label_to_onehot(torch.Tensor(gt_labels).long(), num_classes=num_cls)
    # gt_data = np.stack(gt_data)
    return SimpleTwoPartyDataset(gt_data_a, gt_data_b, gt_labels)


def get_dataset(dataset_name: str, attack_data_type: str):
    if dataset_name == "mnist":
        print("[INFO] Get MNIST dataset.")
        num_classes = 10
        num_samples = 60000
        num_train = 30000
        train_set, test_set, image_half_dim = get_mnist()
        print("train_set:", len(train_set))
        print("test_set:", len(test_set))
        print("image_half_dim:", image_half_dim)
        train_indices = np.random.choice(a=num_samples, size=num_train, replace=False)
    elif dataset_name == "fmnist":
        print("[INFO] Get Fashion-MNIST dataset.")
        num_classes = 10
        num_samples = 60000
        num_train = 30000
        train_set, test_set, image_half_dim = get_fashionmnist()
        print("train_set:", len(train_set))
        print("test_set:", len(test_set))
        print("image_half_dim:", image_half_dim)
        train_indices = np.random.choice(a=num_samples, size=num_train, replace=False)
    elif dataset_name == "cifar":
        print("[INFO] Get CIFAR dataset.")
        num_classes = 10
        num_samples = 50000
        num_train = 30000
        train_set, test_set, image_half_dim = get_cifar()
        train_indices = np.random.choice(a=num_samples, size=num_train, replace=False)
    else:
        raise Exception("Does not support for dataset [{}] for now.".format(dataset_name))

    if attack_data_type == "same":
        print("[INFO] Get {} dataset, the same for training vfl.".format(dataset_name.upper()))
        print("[INFO] train_indices shape:{}".format(train_indices.shape))
        print("[INFO] train_indices:{}".format(train_indices[:50]))
    elif attack_data_type == "iid":
        print("[INFO] Get {} dataset, from the same distribution (IID) as the vfl training.".format(dataset_name.upper()))
        train_indices = np.setdiff1d(np.arange(num_samples), train_indices, assume_unique=True)
        print("[INFO] train_indices shape:{}".format(train_indices.shape))
    else:
        raise Exception("Does not support for attack data type: [{}] for now.".format(attack_data_type))

    train_set = DatasetSplit(train_set, train_indices)
    return train_set, test_set, test_set, num_classes, image_half_dim
    # if dataset == "mnist":
    #     train_set, test_set, image_half_dim = get_mnist()
    #     num_classes = 10
    #     if attack_data_type == "same":
    #         print("[INFO] Get MNIST dataset, which is the same as the vfl training.")
    #         train_indices = np.random.choice(a=60000, size=30000, replace=False)
    #         print("[INFO] train_indices shape:{}".format(train_indices.shape))
    #         print("[INFO] train_indices:{}".format(train_indices[:50]))
    #     elif attack_data_type == "iid":
    #         print("[INFO] Get MNIST dataset, which is from the same distribution (IID) as the vfl training.")
    #         same_indices = np.random.choice(a=60000, size=30000, replace=False)
    #         train_indices = np.setdiff1d(np.arange(60000), same_indices, assume_unique=True)
    #         print("[INFO] train_indices shape:{}".format(train_indices.shape))
    #     else:
    #         raise Exception("Does not support for attack data type: [{}] for now.".format(attack_data_type))
    #     train_set = DatasetSplit(train_set, train_indices)
    #     return train_set, test_set, test_set, num_classes, image_half_dim
    # else:
    #     raise Exception("Does not support for dataset [{}] for now.".format(dataset))


def get_mnist():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)
         ])
    half_dim = 14
    train_dst = datasets.MNIST("~/.torch", download=True, train=True, transform=transform)
    test_dst = datasets.MNIST("~/.torch", download=True, train=False, transform=transform)
    return train_dst, test_dst, half_dim


def get_fashionmnist():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)
         ])
    half_dim = 14
    train_dst = datasets.FashionMNIST("~/.torch", download=True, train=True, transform=transform)
    test_dst = datasets.FashionMNIST("~/.torch", download=True, train=False, transform=transform)
    return train_dst, test_dst, half_dim


def get_cifar():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)
         ])
    half_dim = 16
    train_dst = datasets.CIFAR10("~/.torch", download=True, train=True, transform=transform)
    test_dst = datasets.CIFAR10("~/.torch", download=True, train=False, transform=transform)
    return train_dst, test_dst, half_dim


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        # self.targets = self.dataset.targets

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
