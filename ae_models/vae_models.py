import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from ae_models.autoencoder import initialize_weights


class Encoder(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=64):
        super(Encoder, self).__init__()
        self.z_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * 2)
        )
        initialize_weights(self)

    def forward(self, x):
        z = self.encoder(x)
        encoding = z.view(len(z), -1)

        # The stddev output is treated as the log of the variance of the normal
        # distribution by convention and for numerical stability
        return encoding[:, :self.z_dim], encoding[:, self.z_dim:].exp()

    # def load_model(self, model_full_name):
    #     self.load_state_dict(torch.load(model_full_name))
    #
    # def save_model(self, model_full_name):
    #     torch.save(self.state_dict(), model_full_name)


class Decoder(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=64):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )
        initialize_weights(self)

    def forward(self, x):
        return self.decoder(x)

        # d_y = F.softmax(z, dim=1)
        # d_y = sharpen(d_y, T=1.0)
        # return self.decoder(d_y), d_y
        # return self.decoder(z), d_y

    # def load_model(self, model_full_name):
    #     self.load_state_dict(torch.load(model_full_name))
    #
    # def save_model(self, model_full_name):
    #     torch.save(self.state_dict(), model_full_name)


class VAE(nn.Module):
    """
    VAE Class
    Values:
    z_dim: the dimension of the noise vector, a scalar
    im_chan: the number of channels of the output image, a scalar
            MNIST is black-and-white, so that's our default
    hidden_dim: the inner dimension, a scalar
    """

    def __init__(self, z_dim=32, input_dim=1, hidden_dim=64):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(input_dim, z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, input_dim, hidden_dim)

    def forward(self, input):
        """
        Function for completing a forward pass of the Decoder: Given a noise vector,
        returns a generated image.
        Parameters:
        images: an image tensor with dimensions (batch_size, im_chan, im_height, im_width)
        Returns:
        decoding: the autoencoded image
        q_dist: the z-distribution of the encoding
        """
        q_mean, q_stddev = self.encoder(input)
        q_dist = Normal(q_mean, q_stddev)
        z_sample = q_dist.rsample()  # Sample once from each distribution, using the `rsample` notation
        s_z_sample = F.softmax(z_sample, dim=1)
        decoding = self.decoder(s_z_sample)
        return decoding, q_dist, s_z_sample

    def save_model(self, model_full_name):
        torch.save(self.state_dict(), model_full_name)

    def load_model(self, model_full_name):
        self.load_state_dict(torch.load(model_full_name))
