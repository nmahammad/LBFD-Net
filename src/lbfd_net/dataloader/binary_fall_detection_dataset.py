from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from lbfd_net.helpers.constants import CLASS_LABELS, NormalizationType, SubsetName
from lbfd_net.helpers.create_transforms import create_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BinaryFallDataset(Dataset):
    """
    Dataset class for binary fall detection using lazy PNG loading.

    Expected directory structure:

        root/
        ├── train/
        │     ├── fall/
        │     └── no_fall/
        ├── validation/
        │     ├── fall/
        │     └── no_fall/
        └── test/
              ├── fall/
              └── no_fall/
    """

    def __init__(
        self,
        root_directory_path: Path | str,
        subset: SubsetName,
        use_augmentation: bool = False,
        normalization_type: NormalizationType = "rgb",
        image_size: tuple[int, int] | None = None,
    ):
        self.root_directory_path = Path(root_directory_path)
        self.subset = subset
        self.use_augmentation = use_augmentation
        self.normalization_type = normalization_type

        # Obtain transforms from helper module
        self.convert_to_grayscale, self.image_transforms = create_transforms(
            use_augmentation=self.use_augmentation,
            normalization_type=self.normalization_type,
            image_size=image_size
        )

        # These lists are populated on load, lazy loading
        self.image_file_paths: list[Path] = []
        self.class_labels: list[int] = []

        self.is_data_loaded = False


    def load_dataset(self) -> None:
        """Scan the directory and collect PNG paths + labels."""

        if self.is_data_loaded == True:
            return None

        subset_directory = self.root_directory_path / self.subset

        if not subset_directory.exists():
            raise FileNotFoundError(f"Subset directory not found: {subset_directory}")

        for class_name, class_label in CLASS_LABELS.items():
            class_directory = subset_directory / class_name

            if not class_directory.exists():
                raise FileNotFoundError(
                    f"Class directory '{class_name}' not found at: {class_directory}"
                )

            png_files = list(class_directory.rglob("*.png"))
            self.image_file_paths.extend(png_files)
            self.class_labels.extend([class_label] * len(png_files))

        if not self.image_file_paths:
            raise ValueError(f"No PNG images found in subset '{self.subset}'")

        self._sort_files_and_labels()
        self.is_data_loaded = True


    def _sort_files_and_labels(self) -> None:
        """Sort file paths and labels together for reproducibility."""

        paired_images_and_labels = list(zip(self.image_file_paths, self.class_labels))

        paired_images_and_labels.sort(key=lambda pair: str(pair[0]).lower())

        sorted_images = []
        sorted_labels = []

        for image, label in paired_images_and_labels:
            sorted_images.append(image)
            sorted_labels.append(label)

        self.image_file_paths = sorted_images
        self.class_labels = sorted_labels


    def _ensure_loaded(self) -> None:
        if not self.is_data_loaded:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")


    def __len__(self) -> int:
        self._ensure_loaded()
        dataset_length = len(self.image_file_paths)
        return dataset_length


    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Load and transform a single image."""

        self._ensure_loaded()

        image_path = self.image_file_paths[index]
        label_value = self.class_labels[index]

        with Image.open(image_path) as image:

            if self.convert_to_grayscale:
                image = image.convert("L")
            else:
                image = image.convert("RGB")

            tensor_image = self.image_transforms(image)

        tensor_label = torch.tensor(label_value, dtype=torch.long)

        return tensor_image, tensor_label


    def get_data_loader(
        self,
        batch_size,
        shuffle_data,
        number_of_workers: int = 2,
        prefetch_factor: int = 2,
        pin_memory: bool | None = None,
    ) -> DataLoader:

        if pin_memory is None:
            pin_memory = torch.cuda.is_available()

        if not self.is_data_loaded:
            self.load_dataset()

        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle_data,
            num_workers=number_of_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor
        )


    def get_class_distribution(self) -> dict[str, int]:
        """Return number of fall vs no_fall samples."""
        self._ensure_loaded()

        class_counts: dict[str, int] = {}
        for class_name, class_label in CLASS_LABELS.items():
            count = 0
            for label in self.class_labels:
                if label == class_label:
                    count += 1
            class_counts[class_name] = count

        return class_counts


    def __repr__(self) -> str:

        if self.is_data_loaded:
            size = len(self.image_file_paths)
        else:
            size = "Dataset Not loaded"

        return (
            f"BinaryFallDataset("
            f"subset='{self.subset}', "
            f"augmentation={self.use_augmentation}, "
            f"normalization_type='{self.normalization_type}', "
            f"convert_to_grayscale={self.convert_to_grayscale}, "
            f"loaded={self.is_data_loaded}, "
            f"size={size})"
        )
