from typing import Final, Literal


CLASS_LABELS: Final[dict[str, int]] = {
    "no_fall": 0,
    "fall": 1
}

SUPPORTED_IMAGE_EXTENSIONS: Final[tuple[str]] = (".png",)

DATASET_MEAN_RGB: Final[list[float]] = [0.5288, 0.5161, 0.4727]
DATASET_STD_RGB:  Final[list[float]] = [0.2366, 0.2398, 0.2436]

DATASET_MEAN_GRAY: Final[list[float]] = [0.5]
DATASET_STD_GRAY:  Final[list[float]] = [0.25]

DEFAULT_RGB_IMAGE_SIZE: Final[tuple[int, int]] = (224, 224)
DEFAULT_GRAY_IMAGE_SIZE: Final[tuple[int, int]] = (32, 32)

MODEL_WEIGHTS_ROOT_DIRECTORY = "model_weights"
DATASET_DIRECTORY_PATH = "dataset"

SUBSET_NAME = Literal["train", "validation", "test"]
NORMALIZATION_TYPE = Literal["rgb", "grayscale"]

ModelNameLiteral = Literal["lenet", "alexnet", "lbfdnet"]


EXPERIMENT_NAME = "binary_fall_detection_experiment"
MLFLOW_TRACKING_URI = "mlruns"

SEED = 42