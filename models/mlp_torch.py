import torch.nn as nn


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.0, 0.0)
    # if hasattr(m, "bias"):
    #     m.bias.data.uniform_(-0.0, 0.0)


class LinearModel(nn.Module):

    def __init__(self, input_dim, output_dim=10, bias=True):
        super(LinearModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=bias)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class ActiveLinearModel(nn.Module):
    def __init__(self, dims: list):
        super(ActiveLinearModel, self).__init__()
        self.layers = nn.ModuleList()
        input_size = dims[0]
        self.layers.append(nn.Linear(input_size, dims[-1], bias=True))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PassiveLinearModel(nn.Module):
    def __init__(self, dims: list):
        super(PassiveLinearModel, self).__init__()
        self.layers = nn.ModuleList()
        input_size = dims[0]
        self.layers.append(nn.Linear(input_size, dims[-1], bias=False))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def get_active_linear_model(struct_config):
    struct_config = {} if struct_config is None else struct_config
    layer_input_dim_list = struct_config["layer_input_dim_list"]
    model = PassiveLinearModel(dims=layer_input_dim_list)
    return model


def get_passive_linear_model(struct_config):
    struct_config = {} if struct_config is None else struct_config
    layer_input_dim_list = struct_config["layer_input_dim_list"]
    model = PassiveLinearModel(dims=layer_input_dim_list)
    return model


class ActiveInteractiveModel(nn.Module):
    def __init__(self, dims: list, final_act_type="relu"):
        super(ActiveInteractiveModel, self).__init__()
        self.final_act_type = final_act_type
        self.layers = nn.ModuleList()
        input_size = dims[0]

        self.layers.append(nn.Linear(input_size, dims[-1], bias=True))
        if self.final_act_type == 'sigmoid':
            self.layers.append(nn.Sigmoid())
        elif self.final_act_type == 'relu':
            self.layers.append(nn.ReLU(inplace=True))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PassiveInteractiveModel(nn.Module):
    def __init__(self, dims: list, final_act_type="relu"):
        super(PassiveInteractiveModel, self).__init__()
        self.final_act_type = final_act_type
        self.layers = nn.ModuleList()
        input_size = dims[0]

        self.layers.append(nn.Linear(input_size, dims[-1], bias=False))
        if self.final_act_type == 'sigmoid':
            self.layers.append(nn.Sigmoid())
        elif self.final_act_type == 'relu':
            self.layers.append(nn.ReLU(inplace=True))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP2(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super(MLP2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(inplace=True)
        )

        # torch.nn.init.xavier_uniform_(self.layer1[1].weight)
        # torch.nn.init.zeros_(self.layer1[1].bias)

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim, bias=True),
            # nn.ReLU(inplace=True)
        )
        # torch.nn.init.xavier_uniform_(self.layer2[0].weight)
        # torch.nn.init.zeros_(self.layer2[0].bias)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class MLP1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, output_dim, bias=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        return x


class MLP(nn.Module):
    def __init__(self, dims: list, act_type="sigmoid", final_act_type=None):
        super(MLP, self).__init__()
        input_size = dims[0]
        for idx in range(len(dims)):
            if dims[idx] is None:
                raise Exception("None in {} is not allowed.".format(dims))

        self.layers = nn.ModuleList()
        self.act_type = act_type
        self.final_act_type = final_act_type

        if len(dims) > 2:
            for dim_idx in range(1, len(dims) - 1):
                self.layers.append(nn.Linear(input_size, dims[dim_idx]))
                input_size = dims[dim_idx]  # For the next layer
                # self.layers.append(nn.BatchNorm1d(dims[dim_idx]))
                # self.layers.append(nn.LeakyReLU(inplace=True))
                if self.act_type == 'sigmoid':
                    self.layers.append(nn.Sigmoid())
                elif self.act_type == 'relu':
                    self.layers.append(nn.ReLU(inplace=True))
                elif self.act_type == 'leakyrelu':
                    self.layers.append(nn.LeakyReLU(inplace=True))

        # === the final layer ===
        self.layers.append(nn.Linear(input_size, dims[-1]))
        if self.final_act_type == 'sigmoid':
            self.layers.append(nn.Sigmoid())
        elif self.final_act_type == 'relu':
            self.layers.append(nn.ReLU(inplace=True))
        elif self.final_act_type == 'leakyrelu':
            self.layers.append(nn.LeakyReLU(inplace=True))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def get_active_interactive_model(struct_config):
    struct_config = {} if struct_config is None else struct_config
    layer_input_dim_list = struct_config["layer_input_dim_list"]
    final_act_type = struct_config.get("final_act_type")
    model = ActiveInteractiveModel(dims=layer_input_dim_list, final_act_type=final_act_type)
    # print("[DEBUG] Active interactive model struct: \n {}".format(model))
    return model


def get_passive_interactive_model(struct_config):
    struct_config = {} if struct_config is None else struct_config
    layer_input_dim_list = struct_config["layer_input_dim_list"]
    final_act_type = struct_config.get("final_act_type")
    model = PassiveInteractiveModel(dims=layer_input_dim_list, final_act_type=final_act_type)
    # print("[DEBUG] Passive interactive model struct: \n {}".format(model))
    return model


def get_mlp(struct_config):
    struct_config = {} if struct_config is None else struct_config
    layer_input_dim_list = struct_config["layer_input_dim_list"]
    act_type = struct_config.get("act_type")
    final_act_type = struct_config.get("final_act_type")
    model = MLP(dims=layer_input_dim_list, act_type=act_type, final_act_type=final_act_type)
    # print("[DEBUG] MLP model struct: \n {}".format(model))
    return model
