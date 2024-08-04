import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.simple_dataset import SimpleDataset
from models.mlp_torch import LinearModel, weights_init
from privacy_attack_during_training.RR_util import compute_estimated_labels


def optimize_residue(data_loader, inpupt_dim, learning_rate=0.01, epoch=5):
    mse_criterion = nn.MSELoss()
    residue_model = LinearModel(input_dim=inpupt_dim, output_dim=1, bias=False)
    residue_model.apply(weights_init)

    optimizer = torch.optim.Adam(residue_model.parameters(), lr=learning_rate)

    # tqdm_train = tqdm(range(epoch), desc='Training')
    # for ep in tqdm_train:

    for ep in range(epoch):
        loss_list = list()
        for data, target in data_loader:
            optimizer.zero_grad()

            logit = residue_model(data)
            mse_loss = mse_criterion(logit, target)
            mse_loss.backward()

            optimizer.step()
            loss_list.append(mse_loss.item())
        # tqdm_train.set_postfix({"loss": np.mean(loss_list)})
        # if ep % 100 == 0:
        #     print("ep: {}, loss: {}".format(ep, np.mean(loss_list)))

    return list(residue_model.parameters())[-1].detach().numpy()


def compute_model_grad(residue_d, data):
    grad = 0
    for d, x in zip(residue_d, data):
        grad += d * x
    return grad


def residue_reconstruction_attack(residue_d, attack_assist_args):
    """

    """
    passive_data = attack_assist_args['passive_data']
    passive_data = torch.tensor(passive_data)
    residue_d = torch.tensor(residue_d)

    # === each party computes gradient of local model
    passive_grad = compute_model_grad(residue_d, passive_data)

    # === passive party perform residual reconstruction attack
    # === assuming passive party only knows the batch-sum passive grad
    passive_data_trans = torch.transpose(passive_data, 0, 1)
    input_dim = passive_data_trans.shape[-1]
    passive_feat_dim = passive_data.shape[-1]
    residue_label = passive_grad.reshape(passive_feat_dim, -1)

    # === prepare dataset for residue reconstruction
    rr_dataset = SimpleDataset(passive_data_trans.detach().numpy(), residue_label.detach().numpy(),
                               label_type=torch.float)

    bs = 32
    lr = 0.002
    ep = 1000

    data_loader = DataLoader(rr_dataset, batch_size=bs, shuffle=True)
    estimated_residue = optimize_residue(data_loader=data_loader, inpupt_dim=input_dim, learning_rate=lr, epoch=ep)

    ################################
    # Compute label recovery result
    ################################
    true_residue = residue_d.detach().numpy().flatten()
    estimated_residue = estimated_residue.flatten()
    estimated_labels = compute_estimated_labels(estimated_residue)
    return estimated_labels
