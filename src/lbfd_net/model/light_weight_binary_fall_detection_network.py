import torch.nn as nn
import torch
from .residual_block import ResidualBlock

class LightWeightBinaryFallDetectionNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Initial convolutional feature extraction block
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=3, 
                out_channels=32,
                kernel_size=5, 
                stride=2, 
                padding=2
                ),

            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=3, 
                stride=2,
                padding=1
                )
        )

        # Residual feature extraction blocks
        self.first_residual_block = ResidualBlock(
            input_channels=32, 
            output_channels=64
            )
        
        self.second_residual_block = ResidualBlock(
            input_channels = 64, 
            output_channels=96
            )
        
        # Global average pooling to reduce spatial dimension to 1×1
        self.global_average_pooling = nn.AdaptiveAvgPool2d(
            output_size=1
            )
        
        # Final classifier for binary fall detection
        self.classifier = nn.Linear(
            in_features=96, 
            out_features=1, 
            bias=True)


    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        extracted_features = self.feature_extractor(input_tensor)
        first_residual_block_output = self.first_residual_block(extracted_features)
        second_residual_block_output = self.second_residual_block(first_residual_block_output)
        pooled_output = self.global_average_pooling(second_residual_block_output)
        flattened_output = torch.flatten(pooled_output, start_dim=1)
        classifier_output = self.classifier(flattened_output)

        return classifier_output


