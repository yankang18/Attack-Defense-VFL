import logging
import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils
import torchvision.transforms as transforms
from PIL import Image
import argparse

tp = transforms.ToTensor()


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        cudnn.benchmark = True
        cudnn.enabled = True
        torch.cuda.manual_seed_all(seed)


def keep_predict_loss(y_true, y_pred):
    # print("y_true:", y_true)
    # print("y_pred:", y_pred[0][:5])
    # print("y_true * y_pred:", (y_true * y_pred))
    return torch.sum(y_true * y_pred)


def label_to_one_hot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def sharpen(probabilities, T):
    if probabilities.ndim == 1:
        # print("here 1")
        tempered = torch.pow(probabilities, 1 / T)
        tempered = (
                tempered
                / (torch.pow((1 - probabilities), 1 / T) + tempered)
        )

    else:
        # print("here 2")
        tempered = torch.pow(probabilities, 1 / T)
        tempered = tempered / tempered.sum(dim=-1, keepdim=True)

    return tempered


def get_rand_batch(seed, class_num, batch_size, transform=None):
    path = './data/mini-imagenet/ok/'
    random.seed(seed)

    total_class = os.listdir(path)
    sample_class = random.sample(total_class, class_num)
    num_per_class = [batch_size // class_num] * class_num
    num_per_class[-1] += batch_size % class_num
    img_path = []
    labels = []

    for id, item in enumerate(sample_class):
        img_folder = os.path.join(path, item)
        img_path_list = [os.path.join(img_folder, img).replace('\\', '/') for img in os.listdir(img_folder)]
        sample_img = random.sample(img_path_list, num_per_class[id])
        img_path += sample_img
        labels += ([item] * num_per_class[id])
    img = []
    for item in img_path:
        x = Image.open(item)
        if transform is not None:
            x = transform(x)
        img.append(x)
    return img, labels


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def printf(str, verbose=True):
    if verbose:
        print(str)
# def img_show(img):
#     plt.imshow(img.permute(1, 2, 0).detach().numpy())
#     plt.show()


# def draw_line_chart(name, x, y):
#     '''
#     draw the line chart
#     :param name: list, eg:['sin', 'cos']
#     :param x: list, eg:[[1, 2, 3], [3, 4, 5]]
#     :param y: list, eg:[[1, 2, 3], [3, 4, 5]]
#     :return:
#     '''
#     for i in range(len(x)):
#         plt.plot(x[i], y[i], marker='.', mec='r', mfc='w', label=name[i])
#     plt.legend()  # 让图例生效
#     # plt.xticks(x, names, rotation=45)
#     plt.margins(0)
#     plt.subplots_adjust(bottom=0.15)
#     plt.xlabel("label x")  # X轴标签
#     plt.ylabel("label y")  # Y轴标签
#     plt.title("title")  # 标题
#     plt.show()


def cross_entropy_for_one_hot(pred, target, reduce="mean"):
    # print("pred shape:", pred.shape)
    # print("target shape:", target.shape)
    # return torch.sum(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))
    if reduce == "mean":
        return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))
    elif reduce == "sum":
        return torch.sum(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))
    else:
        raise Exception("Does not support reduce [{}]".format(reduce))


def cross_entropy_for_onehot_samplewise(pred, target):
    return - target * F.log_softmax(pred, dim=-1)


def get_class_i(dataset, label_set):
    gt_data = []
    gt_labels = []
    num_cls = len(label_set)
    for j in range(len(dataset)):
        img, label = dataset[j]
        if label in label_set:
            label_new = label_set.index(label)
            gt_data.append(img if torch.is_tensor(img) else tp(img))
            gt_labels.append(label_new)
            # gt_labels.append(label_to_onehot(torch.Tensor([label_new]).long(),num_classes=num_cls))
    gt_labels = label_to_onehot(torch.Tensor(gt_labels).long(), num_classes=num_cls)
    gt_data = torch.stack(gt_data)
    return gt_data, gt_labels


def append_exp_res(path, res):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(res + '\n')


def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img


def get_timestamp():
    return int(datetime.utcnow().timestamp())


def entropy(predictions):
    epsilon = 1e-6
    H = -predictions * torch.log(predictions + epsilon)
    # print("H:", H.shape)
    return torch.mean(H)


def numpy_entropy(predictions, N=2):
    # epsilon = 1e-10
    # epsilon = 1e-8
    epsilon = 0
    # print(np.log2(predictions + epsilon))
    H = -predictions * (np.log(predictions + epsilon) / np.log(N))
    # print("H:", H.shape)
    return np.sum(H)
    # return H


def calculate_entropy(matrix, N=2):
    class_counts = np.zeros(matrix.shape[0])
    all_counts = 0
    for row_idx, row in enumerate(matrix):
        for elem in row:
            class_counts[row_idx] += elem
            all_counts += elem

    # print("class_counts", class_counts)
    # print("all_counts", all_counts)

    weight_entropy = 0.0
    for row_idx, row in enumerate(matrix):
        norm_elem_list = []
        class_count = class_counts[row_idx]
        for elem in row:
            if elem > 0:
                norm_elem_list.append(elem / float(class_count))
        weight = class_count / float(all_counts)
        # weight = 1 / float(len(matrix))
        ent = numpy_entropy(np.array(norm_elem_list), N=N)
        # print("norm_elem_list:", norm_elem_list)
        # print("weight:", weight)
        # print("ent:", ent)
        weight_entropy += weight * ent
    return weight_entropy


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path, mode='w')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def accuracy(output, target, topk=(1,)):
#     """ Computes the precision@k for the specified values of k """
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     # one-hot case
#     if target.ndimension() > 1:
#         target = target.max(1)[1]
#
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#     res = []
#     for k in topk:
#         correct_k = correct[:k].reshape(-1).float().sum(0)
#         res.append(correct_k.mul_(1.0 / batch_size))
#
#     return res


def accuracy2(predict, target):
    batch_size = target.size(0)
    correct = predict.eq(target).sum()
    # print('correct', correct.sum())
    return correct * (100.0 / batch_size)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print('correct', correct)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# def create_exp_dir(path, scripts_to_save=None):
#     os.makedirs(path, exist_ok=True)
#     print('Experiment dir : {}'.format(path))
#
#     if scripts_to_save is not None:
#         os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
#         for script in scripts_to_save:
#             dst_file = os.path.join(path, 'scripts', os.path.basename(script))
#             shutil.copyfile(script, dst_file)


# def save_checkpoint(state, ckpt_dir, is_best=False):
#     os.makedirs(ckpt_dir, exist_ok=True)
#     filename = os.path.join(ckpt_dir, 'extractor_checkpoints.pth.tar')
#     torch.save(state, filename)
#     if is_best:
#         best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
#         shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    get_rand_batch(1, 4, 8)
