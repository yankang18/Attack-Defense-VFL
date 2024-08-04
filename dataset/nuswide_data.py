import os

import pandas as pd


def get_top_k_labels(data_dir, top_k=5):
    data_path = "NUS_WIDE/Groundtruth/AllLabels"
    label_counts = {}
    for filename in os.listdir(os.path.join(data_dir, data_path)):
        file = os.path.join(data_dir, data_path, filename)
        if os.path.isfile(file):
            label = file[:-4].split("_")[-1]
            df = pd.read_csv(os.path.join(data_dir, file))
            df.columns = ['label']
            label_counts[label] = (df[df['label'] == 1].shape[0])
    label_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    print("label_counts:", label_counts)
    selected = [k for (k, v) in label_counts[:top_k]]
    return selected


def get_labeled_data(data_dir, selected_label, n_samples, dtype="Train"):
    # get labels
    data_path = "NUS_WIDE/Groundtruth/TrainTestLabels/"
    dfs = []
    for label in selected_label:
        file = os.path.join(data_dir, data_path, "_".join(["Labels", label, dtype]) + ".txt")
        df = pd.read_csv(file, header=None)
        # print("df shape", df.shape)
        df.columns = [label]
        dfs.append(df)
    data_labels = pd.concat(dfs, axis=1)
    # print(data_labels)
    if len(selected_label) > 1:
        selected = data_labels[data_labels.sum(axis=1) == 1]
    else:
        selected = data_labels
    print("Selected sample size:{}".format(selected.shape))

    # get XA, which are image low level features
    features_path = "NUS_WIDE/NUS_WID_Low_Level_Features/Low_Level_Features"
    # print("data_dir: {0}".format(data_dir))
    # print("features_path: {0}".format(features_path))
    dfs = []
    for file in os.listdir(os.path.join(data_dir, features_path)):
        if file.startswith("_".join([dtype, "Normalized"])):
            df = pd.read_csv(os.path.join(data_dir, features_path, file), header=None, sep=" ")
            df.dropna(axis=1, inplace=True)
            print("b datasets features", len(df.columns))
            dfs.append(df)
    # print(dfs)
    data_XA = pd.concat(dfs, axis=1)
    data_X_image_selected = data_XA.loc[selected.index]
    print("X image shape:", data_X_image_selected.shape)  # 634 columns

    # get XB, which are tags
    tag_path = "NUS_WIDE/NUS_WID_Tags/"
    file = "_".join([dtype, "Tags1k"]) + ".dat"
    tagsdf = pd.read_csv(os.path.join(data_dir, tag_path, file), header=None, sep="\t")
    tagsdf.dropna(axis=1, inplace=True)
    data_X_text_selected = tagsdf.loc[selected.index]
    print("X text shape:", data_X_text_selected.shape)

    if n_samples is None:
        return data_X_image_selected.values[:], data_X_text_selected.values[:], selected.values[:]
    return data_X_image_selected.values[:n_samples], data_X_text_selected.values[:n_samples], selected.values[
                                                                                              :n_samples]
