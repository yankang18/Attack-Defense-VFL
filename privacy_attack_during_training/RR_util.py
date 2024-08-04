import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import make_interp_spline
from sklearn.metrics import roc_auc_score


def estimated_labels_precision(estimated_labels, true_labels):
    estimated_labels = torch.tensor(estimated_labels)
    correct_results_sum = estimated_labels.eq(true_labels).sum().item()

    if sum(true_labels) == 0:
        auc = 0.5
    else:
        auc = roc_auc_score(true_labels, estimated_labels.numpy())
    return correct_results_sum / len(true_labels), auc


def compute_estimated_labels(residue):
    return [1 if r < 0 else 0 for r in residue]


def make_spline(x, y):
    x_y_spline = make_interp_spline(x, y, k=3)

    # Returns evenly spaced numbers
    # over a specified interval.
    x_ = np.linspace(x.min(), x.max(), 50)

    y_ = x_y_spline(x_)
    return x_, y_


def plot_series(metric_records, legend_list, x_values, x_ticks, x_label="", y_label=""):
    plt.rcParams['pdf.fonttype'] = 42

    style_list = ["orchid", "red", "green", "blue", "purple", "peru", "olive", "coral"]

    marker_list = None
    if len(legend_list) == 2:
        style_list = ["r", "b"]
        marker_list = ["o", "s"]

    elif len(legend_list) == 3:
        style_list = ["r", "b", "g"]
        marker_list = ["o", "s", "^"]

    elif len(legend_list) == 4:
        style_list = ["r", "b", "g", "b--"]
        marker_list = ["o", "s", "*", "^"]

    elif len(legend_list) == 5:
        style_list = ["r", "b", "g", "b--", "orchid"]
        marker_list = ["o", "s", "*", "^", "D"]

    elif len(legend_list) == 6:
        # style_list = ["orchid", "r", "g", "b", "purple", "peru", "olive", "coral"]
        style_list = ["r", "g", "b", "r--", "g--", "b--"]
        # style_list = ["r", "r--", "b", "b--", "g", "g--"]
        # style_list = ["m", "r", "g", "b", "c", "y", "k"]
        marker_list = ["o", "s", "*", "^", "D"]

    elif len(legend_list) == 7:
        # style_list = ["m", "r", "g", "b", "r--", "g--", "b--"]
        style_list = ["orchid", "r", "g", "b", "r--", "g--", "b--"]

    elif len(legend_list) == 8:
        style_list = ["r", "b", "g", "k", "r--", "b--", "g--", "k--"]

    elif len(legend_list) == 9:
        style_list = ["r", "r--", "r:", "b", "b--", "b:", "g", "g--", "g:"]

    else:
        raise Exception("Does not support legend with length:{}".format(len(legend_list)))

    legend_size = 13
    markesize = 6

    plt.subplot(111)

    for i, metrics in enumerate(metric_records):
        # print(metrics)
        # print(style_list)
        # print(marker_list)
        # plt.plot(metrics, style_list[i], marker=marker_list[i], markersize=markesize, linewidth=2.0)
        # plt.plot(metrics, style_list[i], markersize=markesize, linewidth=2.0)
        x, y = make_spline(x_values, np.array(metrics))
        plt.plot(x, y, style_list[i], markersize=markesize, linewidth=2.0)

    plt.xticks(np.arange(len(x_ticks)), x_ticks, fontsize=13)
    plt.yticks(fontsize=12)
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=16)
    # plt.title(scenario + " Party A with " + data_type, fontsize=16)
    plt.legend(legend_list, fontsize=legend_size, loc='best')
    # plt.ylim((65))
    plt.show()
