import os
import torch
from PIL import Image
from torchvision import transforms 
from .constants import DATASET_DIRECTORY_PATH

COLOR_MODE = "grayscale"  # Options: "rgb" or "grayscale"


def collect_training_image_file_paths(dataset_root_directory_path):
    """
    Collect all training image file paths from the dataset.
    Expected directory layout:
        dataset_root/train/fall/
        dataset_root/train/no_fall/
    """
    collected_training_image_file_paths = []

    training_root_directory_path = os.path.join(dataset_root_directory_path, "train")
    class_directory_names = ["fall", "no_fall"]

    for class_directory_name in class_directory_names:
        class_directory_path = os.path.join(training_root_directory_path, class_directory_name)

        if not os.path.isdir(class_directory_path):
            continue

        for file_name in os.listdir(class_directory_path):
            file_name_lower = file_name.lower()
            if file_name_lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                full_image_file_path = os.path.join(class_directory_path, file_name)
                collected_training_image_file_paths.append(full_image_file_path)

    return collected_training_image_file_paths


def compute_dataset_mean_and_standard_deviation(image_file_paths, color_mode):
    """
    Compute dataset mean and standard deviation across all training images.
    Formula:
        mean = E[X]
        std  = sqrt(E[X^2] - (E[X])^2)
    """

    if color_mode == "rgb":
        number_of_channels = 3
        pillow_conversion_mode = "RGB"
    else:
        number_of_channels = 1
        pillow_conversion_mode = "L"

    total_pixel_count = 0
    sum_of_pixel_values_per_channel = torch.zeros(number_of_channels)
    sum_of_squared_pixel_values_per_channel = torch.zeros(number_of_channels)

    for image_file_path in image_file_paths:
        image = Image.open(image_file_path).convert(pillow_conversion_mode)
        image_tensor = transforms.ToTensor()(image)

        channel_count, image_height, image_width = image_tensor.shape
        number_of_pixels_in_image = image_height * image_width

        sum_of_pixel_values_per_channel += image_tensor.sum(dim=(1, 2))
        sum_of_squared_pixel_values_per_channel += (image_tensor ** 2).sum(dim=(1, 2))
        total_pixel_count += number_of_pixels_in_image

    mean_per_channel = sum_of_pixel_values_per_channel / total_pixel_count
    mean_of_squared_values_per_channel = sum_of_squared_pixel_values_per_channel / total_pixel_count

    variance_per_channel = mean_of_squared_values_per_channel - (mean_per_channel ** 2)
    standard_deviation_per_channel = torch.sqrt(variance_per_channel)

    return mean_per_channel, standard_deviation_per_channel



if __name__ == "__main__":

    training_image_file_paths = collect_training_image_file_paths(DATASET_DIRECTORY_PATH)
    print("Number of training images found:", len(training_image_file_paths))

    if len(training_image_file_paths) == 0:
        raise RuntimeError("No training images found. Check dataset path and structure.")

    if COLOR_MODE not in ["rgb", "grayscale"]:
        raise ValueError("COLOR_MODE must be either 'rgb' or 'grayscale'")

    mean_values, standard_deviation_values = compute_dataset_mean_and_standard_deviation(
        training_image_file_paths,
        COLOR_MODE
    )

    print("\n")
    print("Dataset Normalization Statistics")
    print("\n")


    if COLOR_MODE == "rgb":
        mean_values_list = [f"{value:.4f}" for value in mean_values.tolist()]
        standard_deviation_list = [f"{value:.4f}" for value in standard_deviation_values.tolist()]

        print("Mean per channel (R, G, B):", mean_values_list)
        print("Standard deviation per channel (R, G, B):", standard_deviation_list)
        print("Usage in transforms.Normalize():")
        print(f"transforms.Normalize(mean=[{', '.join(mean_values_list)}], std=[{', '.join(standard_deviation_list)}])")

    else:
        mean_value = float(mean_values.item())
        standard_deviation_value = float(standard_deviation_values.item())

        print("Mean (grayscale):", f"{mean_value:.4f}")
        print("Standard deviation (grayscale):", f"{standard_deviation_value:.4f}")
        print("Usage in transforms.Normalize():")
        print(f"transforms.Normalize(mean=[{mean_value:.4f}], std=[{standard_deviation_value:.4f}])")
