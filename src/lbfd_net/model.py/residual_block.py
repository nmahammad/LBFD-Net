import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResidualBlock, self).__init__()

        # Depthwise separable convolution for efficiency
        self.first_depthwise_convolution = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=input_channels, 
            kernel_size=3, 
            stride=2,
            padding=1, 
            groups=input_channels
            )
        

        self.first_pointwise_convolution = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=output_channels, 
            kernel_size=1
            )
        
        self.first_batch_normalization = nn.BatchNorm2d(
            num_features=output_channels
            )

        # Second depthwise separable convolution
        self.second_depthwise_convolution = nn.Conv2d(
            in_channels=output_channels, 
            out_channels=output_channels, 
            kernel_size=3, 
            stride=1,
            padding=1,
            groups=output_channels
            )
        
        
        self.second_pointwise_convolution = nn.Conv2d(
           in_channels=output_channels, 
           out_channels=output_channels, 
           kernel_size=1
           )
        

        self.second_batch_normalization = nn.BatchNorm2d(
            num_features=output_channels
            )


        # Skip connection with projection
        self.skip_connection_projection = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels, 
                out_channels=output_channels, 
                kernel_size=1, 
                stride=2
                ),


            nn.BatchNorm2d(
                num_features=output_channels
                )
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        projected_identity_tensor = self.skip_connection_projection(input_tensor)

        # First depthwise separable convolution
        first_depthwise_convolution_output = self.first_depthwise_convolution(input_tensor)
        first_pointwise_convolution_output = self.first_pointwise_convolution(first_depthwise_convolution_output)
        first_batch_normalization_output = self.first_batch_normalization(first_pointwise_convolution_output)
        relu_activation_output = self.relu(first_batch_normalization_output)
        
        # Second depthwise separable convolution pathway
        second_depthwise_convolution_output = self.second_depthwise_convolution(relu_activation_output)
        second_pointwise_convolution_output = self.second_pointwise_convolution(second_depthwise_convolution_output)
        second_batch_normalization_output = self.second_batch_normalization(second_pointwise_convolution_output)
        
        # Residual addition
        residual_added_tensor = second_batch_normalization_output + projected_identity_tensor
        relu_activated_residual_output = self.relu(residual_added_tensor)

        return relu_activated_residual_output