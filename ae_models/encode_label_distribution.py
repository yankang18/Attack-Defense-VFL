import torch
import torchvision.transforms as transforms
from torchvision import datasets

from ae_models.autoencoder import AutoEncoder


def get_loader(dst, batch_size):
    # return torch.utils.data.DataLoader(dst, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    return torch.utils.data.DataLoader(dst, batch_size=batch_size, num_workers=4)


def label_to_one_hot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


if __name__ == '__main__':

    n_epochs = 10
    learning_rate = 0.001

    # df = pd.DataFrame(
    #     columns=['exp_id', 'dataset', 'num_class', "batch_size", "rec_rate", "rec_rate_mean", "rec_rate_std"])

    transform = transforms.Compose(
        [transforms.ToTensor()
         ])

    dim = 10
    model_timestamp = "1628318258"
    encoder = AutoEncoder(input_dim=dim, encode_dim=2 + dim * 6)
    model_name = f"autoencoder_{dim}_{model_timestamp}"
    encoder.load_model(f"./trained_models/{model_name}")
    batch_size = 1024
    dataset = "MNIST"
    train_dst = datasets.MNIST("~/.torch", download=True, train=True, transform=transform)
    train_loader = get_loader(train_dst, batch_size)
    for i, (bt_data, bt_label) in enumerate(train_loader):
        act_one_hot_label = label_to_one_hot(bt_label, 10)
        act_label = torch.argmax(act_one_hot_label, dim=-1)
        enc_label = torch.argmax(encoder.encoder(act_one_hot_label), dim=-1)
        print("-"*20)
        print(act_label)
        print(enc_label)
