from typing import Literal

from lbfd_net.models.LBFD_Net.light_weight_fall_detection_cnn import LightweightFallDetectionCNN
from lbfd_net.models.AlexNet.alex_fall_net import AlexFallNet
from lbfd_net.models.LeNet.le_fall_net import LeFallNet
from lbfd_net.helpers.constants import ModelNameLiteral


def choose_model_and_settings(model_name: ModelNameLiteral):
    model_name = model_name.lower()

    if model_name == "lenet":
        return LeFallNet(), "grayscale", (32, 32)

    if model_name == "alexnet":
        return AlexFallNet(), "rgb", (224, 224)

    if model_name == "lbfdnet":
        return LightweightFallDetectionCNN(), "rgb", (224, 224)

    raise ValueError(f"Invalid model name: {model_name}")
