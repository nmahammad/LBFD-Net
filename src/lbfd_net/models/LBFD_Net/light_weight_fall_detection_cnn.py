import torch.nn as nn
import torch
from .light_weight_residual_block import LightweightResidualBlock

class LightweightFallDetectionCNN(nn.Module):
    def __init__(self):
        super(LightweightFallDetectionCNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.res_block1 = LightweightResidualBlock(32, 64)
        self.res_block2 = LightweightResidualBlock(64, 96)

        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(in_features=96, out_features=1, bias=True)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

