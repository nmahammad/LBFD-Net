"""
Run full training pipeline for LBFD-Net.
All hyperparameters are declared at the top as uppercase constants.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from lbfd_net.dataloader.binary_fall_detection_dataset import BinaryFallDataset
from lbfd_net.model.light_weight_binary_fall_detection_network import LightWeightBinaryFallDetectionNetwork
from lbfd_net.metrics.metrics_calculator import MetricsCalculator
from lbfd_net.training.training_pipeline import TrainingPipeline


# ================================================================
#                     HYPERPARAMETERS 
# ================================================================

DATASET_PATH = "dataset"

BATCH_SIZE = 16
SHUFFLE_DATA = True

NUMBER_OF_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0

SCHEDULER_STEP_SIZE = 4
SCHEDULER_GAMMA = 0.5

EARLY_STOPPING_PATIENCE = 8

EXPERIMENT_NAME = "lbfd_net_experiment"
TRACKING_URI = "mlruns"

# ================================================================
#                             MAIN
# ================================================================

if __name__ == "__main__":

    # Device
    computing_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", computing_device)

    # -------------------- Dataset --------------------
    training_dataset = BinaryFallDataset(
        root_directory_path=DATASET_PATH,
        subset="train"
    )
    validation_dataset = BinaryFallDataset(
        root_directory_path=DATASET_PATH,
        subset="validation"
    )

    training_dataset.load_dataset()
    validation_dataset.load_dataset()

    training_data_loader = training_dataset.get_data_loader(
        batch_size=BATCH_SIZE,
        shuffle_data=SHUFFLE_DATA
    )
    validation_data_loader = validation_dataset.get_data_loader(
        batch_size=BATCH_SIZE,
        shuffle_data=False
    )

    # -------------------- Model + Loss --------------------
    model = LightWeightBinaryFallDetectionNetwork()
    loss_function = nn.BCEWithLogitsLoss()

    # -------------------- Optimizer + Scheduler --------------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=SCHEDULER_STEP_SIZE,
        gamma=SCHEDULER_GAMMA
    )

    # -------------------- Metrics Calculator --------------------
    metrics_calculator = MetricsCalculator()

    # -------------------- Training Pipeline --------------------
    training_pipeline = TrainingPipeline(
        model=model,
        training_data_loader=training_data_loader,
        validation_data_loader=validation_data_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        computing_device=computing_device,
        number_of_epochs=NUMBER_OF_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        metrics_calculator=metrics_calculator,
        experiment_name=EXPERIMENT_NAME
    )

    # -------------------- Run Training --------------------
    training_pipeline.train_full_cycle()
