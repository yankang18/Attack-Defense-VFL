import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def diversity_loss(G1, G2, z1, z2):
    lz = torch.mean(torch.abs(G2 - G1)) / torch.mean(torch.abs(z2 - z1))
    eps = 1 * 1e-5
    G_div = 1 / (lz + eps)
    return G_div


def prior_mean_loss(image, channel_means):
    num_channels = len(channel_means)
    # print("num_channels", num_channels)
    loss = 0
    for i in range(num_channels):
        loss += torch.pow(torch.mean(image[:, i, :, :]) - channel_means[i], 2)
    return loss


def prior_std_loss(image, channel_stds):
    num_channels = len(channel_stds)
    loss = 0
    for i in range(num_channels):
        loss += torch.pow(torch.std(image[:, i, :, :]) - channel_stds[i], 2)
    return loss


def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


def TVloss(x, pow=2):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), pow).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), pow).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size


def l2loss(x):
    return (x ** 2).mean()


# def TVloss2(x):
#     batch_size = x.size()[0]
#     h_x = x.size()[2]
#     w_x = x.size()[3]
#     count_h = _tensor_size(x[:, :, 1:, :])
#     count_w = _tensor_size(x[:, :, :, 1:])
#     h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
#     w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
#     return (h_tv / count_h + w_tv / count_w) / batch_size


def total_variation_loss(image, exp=2):
    bs, c, h, w = image.size()
    tv_h = torch.pow(image[:, :, 1:, :] - image[:, :, :-1, :], exp).sum()
    tv_w = torch.pow(image[:, :, :, 1:] - image[:, :, :, :-1], exp).sum()
    return (tv_h + tv_w) / (bs * c * h * w)


def total_variation_loss2(image, exp=2):
    # bs, c, h, w = image.size()
    tv_h = torch.pow(image[:, :, 1:, :] - image[:, :, :-1, :], exp)
    tv_w = torch.pow(image[:, :, :, 1:] - image[:, :, :, :-1], exp)
    return torch.mean(torch.sum(tv_h, dim=[1, 2, 3]) + torch.sum(tv_w, dim=[1, 2, 3]))


def l2_loss(image):
    bs, c, h, w = image.size()
    l2 = torch.pow((image + 1) / 2, 2).sum()
    return l2 / (bs * c * h * w)


def l2loss(x):
    return (x ** 2).mean()


def l1loss(x):
    return (torch.abs(x)).mean()


def gaussian(window_size, sigma):
    """
    Generates a list of Tensor values drawn from a gaussian distribution with standard
    diviation = sigma and sum of all elements = 1.

    Length of list = window_size
    """
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    # Generate an 1D tensor containing values sampled from a gaussian distribution
    _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)

    # Converting to 2D
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)

    window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

    return window


def calculate_ssim(img1, img2, window_size=11, window=None, size_average=True, full=False):
    # L = val_range  # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

    pad = window_size // 2

    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    # if window is not provided, init one
    if window is None:
        real_size = min(window_size, height, width)  # window should be at least 11x11
        window = create_window(real_size, channel=channels).to(img1.device)

    # calculating the mu parameter (locally) for both images using a gaussian filter
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability
    C1 = 0.01 ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = 0.03 ** 2

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean()
    else:
        ret = ssim_score.mean(1).mean(1).mean(1)

    if full:
        return ret, contrast_metric

    return ret


def calculate_psnr(real_image, inversion_image, max_val=1):
    mse = ((real_image - inversion_image) ** 2).mean()
    mse = mse.cpu().detach().numpy()
    psnr = 10 * np.log10(max_val ** 2 / mse)
    return psnr


def select_image(dataset, label, index):
    count = 0
    for j in range(len(dataset)):
        img, lbl = dataset[j]
        if label == lbl:
            if index == count:
                return img
            count += 1


def plot_pair_images(images, file_path=None, show_fig=False):
    titles = ["original", "generated"]
    plt.figure(figsize=(10, 6))
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        img = images[i]
        img = img.permute(1, 2, 0)
        # print("img: \n{}".format(img))
        img = (img + 1) / 2
        print(f"{i}, img shape:{img.shape}")
        img = img.cpu().detach().numpy()
        plt.imshow(img, cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    if show_fig:
        plt.show()
    if file_path:
        plt.savefig(file_path)
        print("saved figure : {}".format(file_path))
