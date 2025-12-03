from pathlib import Path
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

from lbfd_net.dataloader.binary_fall_detection_dataset import BinaryFallDataset
from lbfd_net.metrics.metrics_calculator import MetricsCalculator
from lbfd_net.training.train_one_epoch import train_one_epoch
from lbfd_net.training.validate_one_epoch import validate_one_epoch
from lbfd_net.helpers.model_selection import choose_model_and_settings
from lbfd_net.helpers.get_run_index import get_next_run_index
from lbfd_net.helpers.constants import DATASET_DIRECTORY_PATH, EXPERIMENT_NAME


class TrainingPipeline:
    """
    A unified training system that:
    - selects the model and device
    - prepares datasets and dataloaders
    - trains and validates every epoch
    - performs MLflow logging
    - handles early stopping
    - saves and restores best model weights
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int,
        number_of_epochs: int,
        learning_rate: float,
        weight_decay: float,
        scheduler_step_size: int,
        scheduler_gamma: float,
        early_stopping_patience: int,
        shuffle_training_data: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.early_stopping_patience = early_stopping_patience
        self.shuffle_training_data = shuffle_training_data

        # Device
        if torch.cuda.is_available():
            self.computing_device = torch.device("cuda")
        else:
            self.computing_device = torch.device("cpu")

        print("Using device:", self.computing_device)

        # Model + settings
        (
            self.model,
            self.normalization_type,
            self.image_size,
        ) = choose_model_and_settings(self.model_name)

        print("Selected model:", self.model_name)
        print("Normalization:", self.normalization_type)
        print("Image size:", self.image_size)

        self.model = self.model.to(self.computing_device)

        # Prepare datasets and dataloaders
        self.training_dataset = BinaryFallDataset(
            root_directory_path=DATASET_DIRECTORY_PATH,
            subset="train",
            use_augmentation=False,
            normalization_type=self.normalization_type,
            image_size=self.image_size,
        )

        self.validation_dataset = BinaryFallDataset(
            root_directory_path=DATASET_DIRECTORY_PATH,
            subset="validation",
            use_augmentation=False,
            normalization_type=self.normalization_type,
            image_size=self.image_size,
        )

        self.training_dataset.load_dataset()
        self.validation_dataset.load_dataset()

        self.training_data_loader = self.training_dataset.get_data_loader(
            batch_size=self.batch_size,
            shuffle_data=self.shuffle_training_data
        )

        self.validation_data_loader = self.validation_dataset.get_data_loader(
            batch_size=self.batch_size,
            shuffle_data=False
        )

        # Loss function
        self.loss_function = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma
        )

        # Metrics
        self.metrics_calculator = MetricsCalculator()

        # Experiment setup
        self.experiment_name = EXPERIMENT_NAME + "_" + self.model_name

        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment(self.experiment_name)

    # ----------------------------------------------------------------------

    def train(self):
        """
        Main function that runs the full training cycle.
        Includes validation at each epoch, early stopping, weight saving, MLflow logging.
        """

        best_validation_loss = float("inf")
        best_model_state = None
        epochs_without_progress = 0

        run_index = get_next_run_index(self.experiment_name)
        run_name = f"run_{run_index}_{self.model_name}"

        run_directory = Path("model_weights") / run_name
        run_directory.mkdir(parents=True, exist_ok=True)
        best_model_path = run_directory / "best_model.pth"

        with mlflow.start_run(run_name=run_name):

            # Log static parameters
            mlflow.log_param("run_index", run_index)
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("epochs", self.number_of_epochs)
            mlflow.log_param("learning_rate", self.learning_rate)
            mlflow.log_param("early_stopping_patience", self.early_stopping_patience)
            mlflow.log_param("scheduler_step_size", self.scheduler_step_size)
            mlflow.log_param("scheduler_gamma", self.scheduler_gamma)

            for epoch_number in range(1, self.number_of_epochs + 1):

                print("\n===============================================")
                print(f"Epoch {epoch_number} / {self.number_of_epochs}")
                print("===============================================")

                # ---------------- Training ----------------
                training_loss = train_one_epoch(
                    model=self.model,
                    dataloader=self.training_data_loader,
                    loss_function=self.loss_function,
                    optimizer=self.optimizer,
                    computing_device=self.computing_device
                )

                # ---------------- Validation ----------------
                (
                    validation_loss,
                    validation_accuracy,
                    validation_precision,
                    validation_recall,
                    validation_f1,
                ) = validate_one_epoch(
                    model=self.model,
                    dataloader=self.validation_data_loader,
                    loss_function=self.loss_function,
                    metrics_calculator=self.metrics_calculator,
                    computing_device=self.computing_device
                )

                # Scheduler update
                self.scheduler.step()

                # MLflow: log metrics
                mlflow.log_metric("training_loss", training_loss, step=epoch_number)
                mlflow.log_metric("validation_loss", validation_loss, step=epoch_number)
                mlflow.log_metric("validation_accuracy", validation_accuracy, step=epoch_number)
                mlflow.log_metric("validation_precision", validation_precision, step=epoch_number)
                mlflow.log_metric("validation_recall", validation_recall, step=epoch_number)
                mlflow.log_metric("validation_f1_score", validation_f1, step=epoch_number)

                # Console summary
                print(
                    f"Training Loss: {training_loss:.4f} | "
                    f"Validation Loss: {validation_loss:.4f}\n"
                    f"Accuracy: {validation_accuracy:.4f}, "
                    f"Precision: {validation_precision:.4f}, "
                    f"Recall: {validation_recall:.4f}, "
                    f"F1 Score: {validation_f1:.4f}"
                )

                # Early stopping + best model saving
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    epochs_without_progress = 0
                    best_model_state = self.model.state_dict()

                    torch.save(best_model_state, best_model_path)
                    mlflow.log_artifact(str(best_model_path))
                    print("New best model saved.")

                else:
                    epochs_without_progress += 1
                    if epochs_without_progress >= self.early_stopping_patience:
                        print("Early stopping triggered.")
                        break

            # Restore best weights
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                print("\nBest model restored into memory.")

            print("\nTraining complete!")
            print("Best model saved at:", best_model_path)
