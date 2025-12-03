import torch.nn as nn
from torch import Tensor
from torchvision.models import alexnet


class AlexFallNet(nn.Module):

    def __init__(self) -> None:
        super(AlexFallNet, self).__init__()

        self.alexnet = alexnet(weights=None)

        # Modify the final classifier layer for binary classification (1 logit)
        self.alexnet.classifier[6] = nn.Linear(4096, 1)  # Output 1 logit for BCEWithLogitsLoss

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Forward pass of AlexFallNet.

        Args:
            input_tensor (Tensor): Batch of images with shape [batch_size, 3, 224, 224].

        Returns:
            Tensor: Model predictions (logits) with shape [batch_size, 1].
        """
        return self.alexnet(input_tensor)