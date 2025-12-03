from torchvision import transforms
from typing import Literal, Tuple
from lbfd_net.helpers.constants import (
    DATASET_MEAN_RGB,
    DATASET_STD_RGB,
    DATASET_MEAN_GRAY,
    DATASET_STD_GRAY,
)


NORMALIZATION_TYPE = Literal["rgb", "grayscale"]

AUGMENTATION_TRANSFORMS = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
])


def create_transforms(
    use_augmentation: bool,
    normalization_type: NORMALIZATION_TYPE,
    image_size: Tuple[int, int]
):
    if normalization_type == "rgb":
        mean = DATASET_MEAN_RGB
        std = DATASET_STD_RGB
        convert_to_gray = False
    else:
        mean = DATASET_MEAN_GRAY
        std = DATASET_STD_GRAY
        convert_to_gray = True

    base = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    if use_augmentation:
        transform = transforms.Compose([AUGMENTATION_TRANSFORMS] + base)
    else:
        transform = transforms.Compose(base)

    return convert_to_gray, transform
