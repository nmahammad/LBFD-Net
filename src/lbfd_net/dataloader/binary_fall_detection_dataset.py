from pathlib import Path
from typing import Final, Literal

import torch
from PIL import Image, ImageFile
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Handle slightly corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Type definitions
SUBSET_NAMES = Literal["train", "validation", "test"]

# Constants
CLASS_LABELS: Final[dict[str, int]] = {"no_fall": 0, "fall": 1}
SUPPORTED_IMAGE_EXTENSIONS: Final[tuple[str]] = (".png",)
DEFAULT_BATCH_SIZE: Final[int] = 16
DEFAULT_NUMBER_OF_WORKERS: Final[int] = 2
DEFAULT_PREFETCH_FACTOR: Final[int] = 2
DEFAULT_IMAGE_SIZE: Final[tuple[int, int]] = (224, 224)

# Base transforms - always applied
BASE_TRANSFORMS = transforms.Compose([
    transforms.Resize(DEFAULT_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5288, 0.5161, 0.4727], std=[0.2366, 0.2398, 0.2436])
    
])



# Augmentation transforms - only applied when use_augmentation=True
AUGMENTATION_TRANSFORMS = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(probability=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
])

def create_transforms(use_augmentation: bool = False) -> transforms.Compose:
    """Create transforms based on whether augmentation is enabled.
    
    Args:
        use_augmentation: If True, applies data augmentation. If False, only basic transforms.
        
    Returns:
        Composed transforms
    """
    if use_augmentation:
        return transforms.Compose([
            AUGMENTATION_TRANSFORMS,
            BASE_TRANSFORMS,
        ])
    else:
        return BASE_TRANSFORMS


class BinaryFallDataset(Dataset):
    """Dataset for binary fall detection with lazy loading.
    
    This dataset implements lazy loading - images are loaded from disk only when
    accessed through __getitem__, saving memory for large datasets.
    
    Expected folder structure:
    root/
    ├── train/
    │   ├── fall/
    │   └── no_fall/
    ├── validation/
    │   ├── fall/
    │   └── no_fall/
    └── test/
        ├── fall/
        └── no_fall/
    """

    def __init__(
        self,
        root_directory_path: Path | str,
        subset: SUBSET_NAMES,
        use_augmentation: bool = False,
        custom_transforms: transforms.Compose | None = None,
    ) -> None:
        """Initialize the dataset.
        
        Args:
            root_directory_path: Path to dataset root directory
            subset: Which subset to use ("train", "validation", or "test")
            use_augmentation: If True, applies data augmentation. If False, only basic transforms.
            custom_transforms: Optional custom transforms. If provided, overrides augmentation setting.
        """
        self.root_directory_path = Path(root_directory_path)
        self.subset = subset
        self.use_augmentation = use_augmentation
        
        # Set transforms based on parameters
        if custom_transforms is not None:
            self.image_transforms = custom_transforms
        else:
            self.image_transforms = create_transforms(use_augmentation)
        
        # Will be populated by load_dataset() - lazy loading approach
        self.image_file_paths: list[Path] = []
        self.class_labels: list[int] = []
        self.is_data_loaded = False

    def load_dataset(self) -> None:
        """Load image file paths and labels from the dataset directory.
        
        This method scans the directory structure but doesn't load actual images
        into memory - that happens lazily in __getitem__.
        """
        if self.is_data_loaded:
            return

        subset_directory = self.root_directory_path / self.subset
        self._validate_directory_exists(subset_directory, "Subset")

        # Load paths and labels for each class
        for class_name, class_label in CLASS_LABELS.items():
            class_directory = subset_directory / class_name
            self._validate_directory_exists(class_directory, f"Class '{class_name}'")
            
            # Only look for PNG files for quality
            png_image_files = list(class_directory.rglob("*.png"))
            self.image_file_paths.extend(png_image_files)
            self.class_labels.extend([class_label] * len(png_image_files))

        if not self.image_file_paths:
            raise ValueError(f"No PNG images found in {subset_directory}")

        # Sort for reproducible results
        self._sort_files_and_labels()
        self.is_data_loaded = True

    def _validate_directory_exists(self, directory_path: Path, description: str) -> None:
        """Check if directory exists and raise informative error if not.
        
        Args:
            directory_path: Path to check
            description: Human-readable description for error message
        """
        if not directory_path.exists():
            raise FileNotFoundError(f"{description} directory not found: {directory_path}")

    def _sort_files_and_labels(self) -> None:
        """Sort file paths and labels together for reproducible ordering.
        
        This ensures that the dataset order is consistent across runs,
        which is important for reproducible training.
        """
        sorted_indices = sorted(
            range(len(self.image_file_paths)),
            key=lambda current_index: str(self.image_file_paths[current_index]).lower()
        )
        self.image_file_paths = [self.image_file_paths[current_index] for current_index in sorted_indices]
        self.class_labels = [self.class_labels[current_index] for current_index in sorted_indices]

    def _ensure_data_loaded(self) -> None:
        """Ensure dataset is loaded before accessing data."""
        if not self.is_data_loaded:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")

    def __len__(self) -> int:
        """Return number of images in dataset."""
        self._ensure_data_loaded()
        return len(self.image_file_paths)

    def __getitem__(self, dataset_index: int) -> tuple[Tensor, Tensor]:
        """Load and transform a single image on demand (lazy loading).
        
        This is where the actual lazy loading happens - images are only loaded
        from disk when requested, saving memory.
        
        Args:
            dataset_index: Index of the sample to retrieve
            
        Returns:
            Tuple of (image_tensor, label_tensor)
        """
        self._ensure_data_loaded()

        image_file_path = self.image_file_paths[dataset_index]
        class_label = self.class_labels[dataset_index]

        # Load image and apply transforms (this is the lazy loading part)
        with Image.open(image_file_path) as loaded_image:
            # Convert to RGB to ensure consistent 3-channel format
            rgb_image = loaded_image.convert("RGB")
            transformed_image_tensor = self.image_transforms(rgb_image)

        # Convert label to tensor for PyTorch
        label_tensor = torch.tensor(class_label, dtype=torch.long)
        
        return transformed_image_tensor, label_tensor

    def get_data_loader(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        shuffle_data: bool = True,
        number_of_workers: int = DEFAULT_NUMBER_OF_WORKERS,
    ) -> DataLoader:
        """Create a DataLoader for this dataset.
        
        Args:
            batch_size: Number of samples per batch
            shuffle_data: Whether to shuffle the dataset
            number_of_workers: Number of worker processes for data loading
            
        Returns:
            Configured DataLoader instance
        """
        if not self.is_data_loaded:
            self.load_dataset()

        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle_data,
            num_workers=number_of_workers,
            pin_memory=torch.cuda.is_available(),  # Faster GPU transfer if available
            prefetch_factor=DEFAULT_PREFETCH_FACTOR,  # Prefetch batches for speed
        )

    def get_class_distribution(self) -> dict[str, int]:
        """Get the distribution of classes in the dataset.
        
        Returns:
            Dictionary mapping class names to counts
        """
        self._ensure_data_loaded()
        
        class_counts = {}
        for class_name, class_label in CLASS_LABELS.items():
            class_count = sum(1 for label in self.class_labels if label == class_label)
            class_counts[class_name] = class_count
        
        return class_counts

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (
            f"BinaryFallDataset("
            f"subset='{self.subset}', "
            f"augmentation={self.use_augmentation}, "
            f"loaded={self.is_data_loaded}, "
            f"size={len(self) if self.is_data_loaded else 'unknown'})"
        )