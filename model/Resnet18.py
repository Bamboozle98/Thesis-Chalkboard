import torch
import torch.nn as nn
import torchvision.models as models


# Freeze the ResNet18 model to see how the transformer performs
class ResNet18(nn.Module):
    def __init__(self, num_classes=512, pretrained=True):
        super(ResNet18, self).__init__()
        # Load pre-trained ResNet18 model
        resnet = models.resnet18(pretrained=pretrained)

        # Remove the fully connected layers (last layers)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Exclude FC layer and avgpool

        # Add a convolutional layer to adjust output channels if needed
        self.conv_out = nn.Conv2d(512, num_classes, kernel_size=1)  # Adjust channels to 512 or any required number

        # Upsample the output to match the input size (224x224)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.features(x)  # Extract feature maps from ResNet
        x = self.conv_out(x)  # Adjust number of channels
        x = self.upsample(x)  # Upsample to 224x224
        return x
