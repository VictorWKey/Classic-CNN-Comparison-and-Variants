import torch.nn
from torch.nn import Module, Conv2d, Linear, MaxPool2d, ReLU, BatchNorm1d, BatchNorm2d, Dropout
from torch import flatten

class CustomLenet_NoBN(Module):
    """CustomLenet sin BatchNorm para comparar el impacto de la normalización"""
    def __init__(self, nChannels, nClasses):
        super(CustomLenet_NoBN, self).__init__()
        self.conv1 = Conv2d(nChannels, 32, kernel_size=5, stride=1, padding=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool = MaxPool2d(kernel_size=2, stride=2)
        self.relu = ReLU()
        self.fc1 = Linear(6*6*64, 256)
        self.dropot_fc1 = Dropout(0.5)
        self.fc2 = Linear(256, 128)
        self.dropot_fc2 = Dropout(0.5)
        self.fc3 = Linear(128, nClasses)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = flatten(x, 1)

        x = self.fc1(x)
        x = self.dropot_fc1(x)

        x = self.fc2(x)
        x = self.dropot_fc2(x)

        x = self.fc3(x)
        return x

class CustomLenet_NoDropout(Module):
    """CustomLenet sin Dropout para comparar el impacto de la regularización"""
    def __init__(self, nChannels, nClasses):
        super(CustomLenet_NoDropout, self).__init__()
        self.conv1 = Conv2d(nChannels, 32, kernel_size=5, stride=1, padding=1)
        self.bn1   = BatchNorm2d(32)
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = BatchNorm2d(64)
        self.maxpool = MaxPool2d(kernel_size=2, stride=2)
        self.relu = ReLU()
        self.fc1 = Linear(6*6*64, 256)
        self.bn3 = BatchNorm1d(256)
        self.fc2 = Linear(256, 128)
        self.bn4 = BatchNorm1d(128)
        self.fc3 = Linear(128, nClasses)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = flatten(x, 1)

        x = self.fc1(x)
        x = self.bn3(x)

        x = self.fc2(x)
        x = self.bn4(x)

        x = self.fc3(x)
        return x