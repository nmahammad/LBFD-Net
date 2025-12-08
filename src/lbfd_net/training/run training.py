from typing import Final
from lbfd_net.training.training_pipeline import TrainingPipeline
from lbfd_net.helpers.set_random_seed import set_random_seed

from lbfd_net.helpers.constants import ModelName, SEED_VALUE

MODEL_NAME: Final[ModelName] = "alexnet"

TRAINING_BATCH_SIZE: Final[int] = 16
SHOULD_SHUFFLE_TRAINING_DATA: Final[bool] = True

TOTAL_NUMBER_OF_EPOCHS: Final[int] = 100
LEARNING_RATE_VALUE: Final[float] = 5e-5
WEIGHT_DECAY_VALUE: Final[float] = 5e-3

SCHEDULER_STEP_SIZE_VALUE: Final[int] = 4
SCHEDULER_GAMMA_VALUE: Final[float] = 0.5

EARLY_STOPPING_PATIENCE_VALUE: Final[int] = 8

def main():

    # set_random_seed(SEED_VALUE)

    training_pipeline_instance = TrainingPipeline(
        model_name=MODEL_NAME,
        batch_size=TRAINING_BATCH_SIZE,
        number_of_epochs=TOTAL_NUMBER_OF_EPOCHS,
        learning_rate=LEARNING_RATE_VALUE,
        weight_decay=WEIGHT_DECAY_VALUE,
        scheduler_step_size=SCHEDULER_STEP_SIZE_VALUE,
        scheduler_gamma=SCHEDULER_GAMMA_VALUE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE_VALUE,
        shuffle_training_data=SHOULD_SHUFFLE_TRAINING_DATA,
    )

    training_pipeline_instance.train()


if __name__ == "__main__":
    main()
