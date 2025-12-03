from lbfd_net.dataloader.binary_fall_detection_dataset import create_transforms

def unpack_transforms(use_augmentation, normalization_type, image_size):
    convert_to_grayscale, composed_transforms = create_transforms(
        use_augmentation=use_augmentation,
        normalization_type=normalization_type,
        image_size=image_size
    )
    return convert_to_grayscale, composed_transforms
