import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, input_dim, num_classes=10):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
