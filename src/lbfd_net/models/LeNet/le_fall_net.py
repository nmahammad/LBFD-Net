import torch
import torch.nn as nn
import torch.nn.functional as F

class LeFallNet(nn.Module):
    def __init__(self, num_channels: int = 1):
        """
        Official LeNet-5 architecture adapted for 32x32 input images and binary classification.

        Args:
            num_channels (int): Number of input channels. Default is 1 (grayscale).
                                For RGB images, set num_channels=3.
        """
        super(LeFallNet, self).__init__()
        # C1: Convolutional layer: input -> 6 feature maps, kernel 5x5
        # For a 32x32 input, output size = 32 - 5 + 1 = 28, so output shape is [batch, 6, 28, 28]
        self.conv1 = nn.Conv2d(num_channels, 6, kernel_size=5, stride=1, padding=0)
        # S2: Average Pooling: kernel size 2, stride 2 => reduces spatial size by half: 28 -> 14
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C3: Convolutional layer: 6 -> 16 feature maps, kernel 5x5
        # Input size: 14x14, output size = 14 - 5 + 1 = 10, shape: [batch, 16, 10, 10]
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        # S4: Average Pooling: kernel size 2, stride 2 => 10 -> 5, shape: [batch, 16, 5, 5]
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # F5: Fully connected layer: Flattened features (16 * 5 * 5 = 400) to 120 units
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # F6: Fully connected layer: 120 to 84 units
        self.fc2 = nn.Linear(120, 84)
        # F7: Fully connected output layer: 84 to 1 unit (for binary classification)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.pool2(x)

        # flatten the features
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = torch.tanh(x)

        x = self.fc2(x)
        x = torch.tanh(x)

        x = self.fc3(x)
        return x
