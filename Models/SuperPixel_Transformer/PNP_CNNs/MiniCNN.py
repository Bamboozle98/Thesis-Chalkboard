import torch.nn as nn
import torch.nn.functional as F


class SuperpixelCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=512):
        super(SuperpixelCNN, self).__init__()
        # 5-layer CNN based on discussed example in meeting
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=3, padding=1)
        # self.pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        # x = self.pool(x)  # Pooling to get feature vector
        # x = x.view(x.size(0), -1)  # Flatten to (batch_size, feature_dim)
        return x
