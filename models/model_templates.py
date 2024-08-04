import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 20, (5, 5))
        self.fc1 = nn.Linear(20 * 1 * 5, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 1 * 5)
        x = self.fc1(x)
        x = F.relu(x)
        return x


class MLP2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 32, bias=True),
            nn.ReLU(inplace=True)
        )
        torch.nn.init.xavier_uniform_(self.layer1[1].weight)
        torch.nn.init.zeros_(self.layer1[1].bias)

        self.layer2 = nn.Sequential(
            nn.Linear(32, output_dim, bias=True),
            nn.ReLU(inplace=True)
        )
        # torch.nn.init.xavier_uniform_(self.layer2[0].weight)
        # torch.nn.init.zeros_(self.layer2[0].bias)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class ClassificationModelHost(nn.Module):

    def __init__(self, local_model):  # , hidden_dim, num_classes):
        super().__init__()
        self.local_model = local_model

    def forward(self, input_X):
        z = self.local_model(input_X)  # .flatten(start_dim=1)
        return z

    # def get_prediction(self, z0, z_list):
    #     if z_list is not None:
    #         out = z0
    #         for item in z_list:
    #             out = out + item
    #     else:
    #         out = z0
    #     return out

    def get_prediction(self, z0, z1):
        return z0 + z1

    def load_local_model(self, load_path, device):
        self.local_model.load_state_dict(torch.load(load_path, map_location=device))


class ClassificationModelHostHead(nn.Module):

    def __init__(self):  # , hidden_dim, num_classes):
        super().__init__()

    def forward(self, z0, z1):
        out = z0.add(z1)
        return out


class ClassificationModelHostHeadWithSoftmax(nn.Module):

    def __init__(self):  # , hidden_dim, num_classes):
        super().__init__()
        self.softmax = nn.LogSoftmax()

    def forward(self, z0, z1):
        out = z0.add(z1)
        return self.softmax(out)


class ClassificationModelHostTrainable(nn.Module):

    def __init__(self, local_model, hidden_dim, num_classes):
        super().__init__()
        self.local_model = local_model
        self.classifier_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_X):
        z = self.local_model(input_X).flatten(start_dim=1)
        return z

    # def get_prediction(self, z0, z_list):
    #     if z_list is not None:
    #         out = torch.cat([z0] + z_list, dim=1)
    #     else:
    #         out = z0
    #     return self.classifier_head(out)

    def get_prediction(self, z0, z1):
        out = torch.cat([z0, z1], dim=1)
        return self.classifier_head(out)


class ClassificationModelHostTrainableHead(nn.Module):

    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.classifier_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, z0, z1):
        out = torch.cat([z0, z1], dim=1)
        return self.classifier_head(out)

    # def get_prediction(self, z0, z_list):
    #     if z_list is not None:
    #         out = torch.cat([z0] + z_list, dim=1)
    #     else:
    #         out = z0
    #     return self.classifier_head(out)


class ClassificationModelHostHeadTrainable(nn.Module):

    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.classifier_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, z0, z1):
        out = torch.cat([z0, z1], dim=1)
        return self.classifier_head(out)


class ClassificationModelGuest(nn.Module):

    def __init__(self, local_model):  # ), hidden_dim, num_classes):
        super().__init__()
        self.local_model = local_model

    def forward(self, input_X):
        z = self.local_model(input_X).flatten(start_dim=1)
        return z

    def load_local_model(self, load_path, device):
        self.local_model.load_state_dict(torch.load(load_path, map_location=device))


class ClassificationModelHost_MLP3(nn.Module):

    def __init__(self, local_model, hidden_dims, num_classes):
        super().__init__()
        self.local_model = local_model
        self.fc1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc2 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc3 = nn.Linear(hidden_dims[2], num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_X):
        x = self.local_model(input_X).flatten(start_dim=1)
        return x

    def get_prediction(self, z_0, z_list):
        if z_list is not None:
            out = torch.cat([z_0] + z_list, dim=1)
        else:
            out = z_0
        x = self.relu(self.fc1(out))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def loss(self, z, label):
        y = self.forward(z)
        loss = F.nll_loss(y, label)
        return loss

    def load_local_model(self, load_path, device):
        self.local_model.load_state_dict(torch.load(load_path, map_location=device))


class ClassificationModelHost_MLP2(nn.Module):

    def __init__(self, local_model, hidden_dims, num_classes):
        super().__init__()
        self.local_model = local_model
        self.fc1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc2 = nn.Linear(hidden_dims[1], num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_X):
        x = self.local_model(input_X).flatten(start_dim=1)
        return x

    def get_prediction(self, z_0, z_list):
        if z_list is not None:
            out = torch.cat([z_0] + z_list, dim=1)
        else:
            out = z_0
        x = self.relu(self.fc1(out))
        x = self.fc2(x)
        return x

    def loss(self, z, label):
        y = self.forward(z)
        loss = F.nll_loss(y, label)
        return loss

    def load_local_model(self, load_path, device):
        self.local_model.load_state_dict(torch.load(load_path, map_location=device))
