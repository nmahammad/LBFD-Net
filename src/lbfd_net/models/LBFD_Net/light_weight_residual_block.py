import torch.nn as nn

class LightweightResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(LightweightResidualBlock, self).__init__()

        # Depthwise separable convolution for efficiency
        self.depthwise = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=2,
                                   padding=1, groups=input_channels)
        self.pointwise = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(output_channels)

        # Second depthwise separable convolution
        self.depthwise2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1,
                                    padding=1, groups=output_channels)
        self.pointwise2 = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(output_channels)

        # Skip connection with projection
        self.skip = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(output_channels)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.depthwise2(out)
        out = self.pointwise2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
