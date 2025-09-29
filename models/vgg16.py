from torch.nn import Module, ReLU, MaxPool2d, Conv2d, Linear, AdaptiveAvgPool2d, Dropout
from torch import flatten

class VGG16(Module):
    def __init__(self, nChannels, nClasses):
        super(VGG16, self).__init__()
        
        # Bloque 1
        self.conv1_1 = Conv2d(nChannels, 64, 3, padding=1)
        self.conv1_2 = Conv2d(64, 64, 3, padding=1)
        
        # Bloque 2
        self.conv2_1 = Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = Conv2d(128, 128, 3, padding=1)
        
        # Bloque 3
        self.conv3_1 = Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = Conv2d(256, 256, 3, padding=1)
        
        # Bloque 4
        self.conv4_1 = Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = Conv2d(512, 512, 3, padding=1)
        
        # Bloque 5
        self.conv5_1 = Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = Conv2d(512, 512, 3, padding=1)
        
        self.maxpool = MaxPool2d(2, stride=2)
        self.relu = ReLU()
        
        self.adaptive_pool = AdaptiveAvgPool2d((7, 7))
        
        self.fc1 = Linear(512 * 7 * 7, 4096)
        self.fc2 = Linear(4096, 4096)
        self.fc3 = Linear(4096, nClasses)
        
        self.dropout = Dropout(0.5)

    def forward(self, x):
        # Bloque 1
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Bloque 2
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Bloque 3
        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.conv3_3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Bloque 4
        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        x = self.conv4_3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Bloque 5
        x = self.conv5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.relu(x)
        x = self.conv5_3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.adaptive_pool(x)
        x = flatten(x, 1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x