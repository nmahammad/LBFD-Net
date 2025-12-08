from typing import Final, Literal


CLASS_LABELS: Final[dict[str, int]] = {
    "no_fall": 0,
    "fall": 1
}

SUPPORTED_IMAGE_EXTENSIONS: Final[tuple[str]] = (".png",)

DATASET_MEAN_RGB: Final[list[float]] = [0.5394, 0.5335, 0.4960]
DATASET_STD_RGB:  Final[list[float]] = [0.2520, 0.2572, 0.2627]

DATASET_MEAN_GRAY: Final[list[float]] = [0.5308]
DATASET_STD_GRAY:  Final[list[float]] = [0.2528]

DEFAULT_RGB_IMAGE_SIZE: Final[tuple[int, int]] = (224, 224)
DEFAULT_GRAY_IMAGE_SIZE: Final[tuple[int, int]] = (32, 32)

MODEL_WEIGHTS_ROOT_DIRECTORY = "model_weights"
DATASET_DIRECTORY_PATH = "dataset"

SubsetName = Literal["train", "validation", "test"]
NormalizationType = Literal["rgb", "grayscale"]
ModelName = Literal["lenet", "alexnet", "lbfdnet"]

EXPERIMENT_NAME = "binary_fall_detection_experiment"
MLFLOW_TRACKING_URI = "mlruns"

SEED_VALUE = 42