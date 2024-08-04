import numpy as np
import torch

from dataset.nuswide_data import get_labeled_data


class NUSWIDEDataset(object):

    def __init__(self, data_dir, selected_labels, data_type):
        self.data_dir = data_dir
        # self.class_num = len(selected_labels)

        if data_type == 'train':
            X_image, X_text, Y = get_labeled_data(self.data_dir, selected_labels, None, 'Train')
        else:
            X_image, X_text, Y = get_labeled_data(self.data_dir, selected_labels, None, 'Test')

        # self.x = [torch.tensor(X_image, dtype=torch.float32), torch.tensor(X_text, dtype=torch.float32)]
        self.x = [torch.tensor(X_text, dtype=torch.float32), torch.tensor(X_image, dtype=torch.float32)]

        # this transforms one hot label to one integer label
        # self.y = torch.tensor(np.argmax(np.array(Y), axis=1), dtype=torch.long)
        # self.y = torch.squeeze(self.y)
        self.y = np.argmax(np.array(Y), axis=1)
        self.y = np.squeeze(self.y)

        print("[Debug] sum y: {}".format(np.sum(self.y)))
        print("[Debug] y: {}".format(self.y))

        # check dataset
        print('[NUSWIDEDataset] data shape:{},{},{}'.format(self.x[0].shape, self.x[1].shape, self.y.shape))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):  # this is single index
        return [self.x[0][index], self.x[1][index]], self.y[index]


if __name__ == '__main__':
    data_dir = "../data/NUS_WIDE"

    # sel = get_top_k_labels(data_dir=data_dir, top_k=10)
    # print("sel", sel)
    # ['sky', 'clouds', 'person', 'water', 'animal', 'grass', 'buildings', 'window', 'plants', 'lake']

    # sel_lbls = get_top_k_labels(data_dir, 81)
    # print(sel_lbls)

    # train_dataset = NUSWIDEDatasetVFLPERROUND(data_dir, 'train')
    # print(train_dataset.y)

    # print(train_dataset.poison_list)
