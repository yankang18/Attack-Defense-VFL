import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, data, labels, label_type=torch.long):
        self.data = data
        self.labels = labels
        self.label_type = label_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item_idx):
        data_i, target_i = self.data[item_idx], self.labels[item_idx]
        if not isinstance(data_i, torch.Tensor):
            data_i = torch.tensor(data_i).float()
        if not isinstance(target_i, torch.Tensor):
            target_i = torch.tensor(target_i, dtype=self.label_type)
        return data_i, target_i


class SimpleTwoPartyDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, data_a, data_b, labels):
        self.data_a = data_a
        self.data_b = data_b
        self.labels = labels

    def __len__(self):
        return len(self.data_a)

    def __getitem__(self, item_idx):
        data_a_i, data_b_i, target_i = self.data_a[item_idx], self.data_b[item_idx], self.labels[item_idx]
        if not isinstance(data_a_i, torch.Tensor):
            data_a_i = torch.tensor(data_a_i).float()
        if not isinstance(data_b_i, torch.Tensor):
            data_b_i = torch.tensor(data_b_i).float()
        if not isinstance(target_i, torch.Tensor):
            target_i = torch.tensor(target_i, dtype=torch.long)
        return (data_a_i, data_b_i), target_i


def get_dataloaders(train_dataset: SimpleTwoPartyDataset, valid_dataset: SimpleTwoPartyDataset, batch_size=32,
                    num_workers=1):
    mnist_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    mnist_valid_loader = None
    if valid_dataset is not None:
        mnist_valid_loader = DataLoader(valid_dataset, batch_size=batch_size * 2, shuffle=True, num_workers=num_workers)
    return mnist_train_loader, mnist_valid_loader
