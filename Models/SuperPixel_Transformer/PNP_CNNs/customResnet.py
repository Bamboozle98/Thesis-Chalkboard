import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

class CustomResNet20(nn.Module):
    def __init__(self, num_classes=512):
        super(CustomResNet20, self).__init__()

        # Initial conv + BN + ReLU + MaxPool
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Layer 1: 2 BasicBlocks (same as ResNet18)
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)

        # Layer 2: 2 BasicBlocks; first block requires downsampling since stride=2 and channels change
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)

        # Layer 3: 3 BasicBlocks (instead of 2 like ResNet18) → ResNet20, with downsampling on the first block
        self.layer3 = self._make_layer(128, 256, num_blocks=3, stride=2)

        # Layer 4: 2 BasicBlocks; first block downsamples from 256 to 512 channels
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        # Final conv layer to output desired channels
        self.conv_out = nn.Conv2d(512, num_classes, kernel_size=1)

        # Upsample back to input resolution
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        # If stride is not 1 or if the number of input channels differs from output channels,
        # create a downsample layer to match dimensions.
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # 3 blocks here → ResNet20
        x = self.layer4(x)
        x = self.conv_out(x)
        x = self.upsample(x)
        return x
