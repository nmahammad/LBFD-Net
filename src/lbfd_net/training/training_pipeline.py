from datetime import datetime
from pathlib import Path
import mlflow
import torch
import torch.nn as nn

from lbfd_net.training.train_one_epoch import train_one_epoch
from lbfd_net.training.validate_one_epoch import validate_one_epoch
from lbfd_net.metrics.metrics_calculator import MetricsCalculator

def get_next_run_index(experiment_name: str) -> int:
    """
    Returns the next run index for a given MLflow experiment name.
    Example: if runs are [run_1_xxx, run_2_xxx], return 3.
    """
    mlflow.set_tracking_uri("mlruns")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return 1

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    indices = []

    for name in runs["tags.mlflow.runName"]:
        if isinstance(name, str) and name.startswith("run_"):
            try:
                index = int(name.split("_")[1])
                indices.append(index)
            except:
                pass

    return max(indices) + 1 if indices else 1


class TrainingPipeline:
    """
    Full training lifecycle controller with MLflow logging, early stopping,
    best-model saving, and learning-rate scheduling.
    """

    def __init__(
        self,
        model: nn.Module,
        training_data_loader: torch.utils.data.DataLoader,
        validation_data_loader: torch.utils.data.DataLoader,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        computing_device: torch.device,
        number_of_epochs: int,
        early_stopping_patience: int,
        metrics_calculator: MetricsCalculator,
        experiment_name: str = "lbfd_net_training_experiment"
    ):
        self.model = model.to(computing_device)
        self.training_data_loader = training_data_loader
        self.validation_data_loader = validation_data_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.computing_device = computing_device
        self.number_of_epochs = number_of_epochs
        self.early_stopping_patience = early_stopping_patience
        self.metrics_calculator = metrics_calculator
        self.experiment_name = experiment_name

        # Configure MLflow local tracking
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment(self.experiment_name)

    # ------------------------------------------------------------------
    def train_full_cycle(self):
        """
        Runs full training lifecycle with:
            - MLflow logging
            - best-model saving
            - early stopping
            - scheduler stepping
            - restoring best weights at the end
        """

        best_validation_loss = float("inf")
        best_model_state = None
        epochs_without_improvement = 0

        # ------------------------------------------------------------------
        # Create a clean experiment directory
        # ------------------------------------------------------------------
        run_index = get_next_run_index(self.experiment_name)
        timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
        run_name = f"run_{run_index}_{timestamp}"
        
        experiment_directory = Path(f"experiments/{self.experiment_name}_{run_name}")
        experiment_directory.mkdir(parents=True, exist_ok=True)

        with mlflow.start_run(run_name=run_name):

            # --------------------- Log Hyperparameters ---------------------
            mlflow.log_param("run_index", run_index)
            mlflow.log_param("timestamp", timestamp)    
            mlflow.log_param("batch_size", self.training_data_loader.batch_size)
            mlflow.log_param("number_of_epochs", self.number_of_epochs)
            mlflow.log_param("learning_rate", self.optimizer.param_groups[0]["lr"])
            mlflow.log_param("early_stopping_patience", self.early_stopping_patience)
            mlflow.log_param("scheduler_step_size", self.scheduler.step_size)
            mlflow.log_param("scheduler_gamma", self.scheduler.gamma)
            mlflow.log_param("experiment_directory", str(experiment_directory))

            print(f"\nStarting training → MLflow run name: {run_name}\n")


            # ========================== MAIN LOOP ==========================
            for epoch_index in range(1, self.number_of_epochs + 1):

                print(f"\n=== Epoch {epoch_index}/{self.number_of_epochs} ===")

                # ---------------------- TRAIN ----------------------
                average_training_loss = train_one_epoch(
                    model=self.model,
                    dataloader=self.training_data_loader,
                    loss_function=self.loss_function,
                    optimizer=self.optimizer,
                    computing_device=self.computing_device
                )

                # -------------------- VALIDATE ---------------------
                (
                    average_validation_loss,
                    validation_accuracy,
                    validation_precision,
                    validation_recall,
                    validation_f1_score,
                ) = validate_one_epoch(
                    model=self.model,
                    dataloader=self.validation_data_loader,
                    loss_function=self.loss_function,
                    metrics_calculator=self.metrics_calculator,
                    computing_device=self.computing_device
                )

                # -------------------- Scheduler --------------------
                self.scheduler.step()

                # -------------------- MLflow Logging --------------------
                mlflow.log_metric("training_loss", average_training_loss, step=epoch_index)
                mlflow.log_metric("validation_loss", average_validation_loss, step=epoch_index)
                mlflow.log_metric("validation_accuracy", validation_accuracy, step=epoch_index)
                mlflow.log_metric("validation_precision", validation_precision, step=epoch_index)
                mlflow.log_metric("validation_recall", validation_recall, step=epoch_index)
                mlflow.log_metric("validation_f1_score", validation_f1_score, step=epoch_index)

                # -------------------- Console Output --------------------
                print(f"Training   - Loss: {average_training_loss:.4f}")
                print(
                    f"Validation - Loss: {average_validation_loss:.4f} | "
                    f"Acc: {validation_accuracy:.4f} | "
                    f"Prec: {validation_precision:.4f} | "
                    f"Rec: {validation_recall:.4f} | "
                    f"F1: {validation_f1_score:.4f}"
                )

                # ==============================================================
                #   EARLY STOPPING + BEST-MODEL SAVING (FULLY FIXED VERSION)
                # ==============================================================
                if average_validation_loss < best_validation_loss:

                    best_validation_loss = average_validation_loss
                    epochs_without_improvement = 0

                    # Save best state *in memory*
                    best_model_state = self.model.state_dict()

                    # Save best to disk
                    best_model_path = experiment_directory / "best_model.pth"
                    torch.save(best_model_state, best_model_path)
                    mlflow.log_artifact(str(best_model_path))

                    print(f"=> Best model updated: {best_model_path}")

                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.early_stopping_patience:
                        print("\nEarly stopping triggered.")
                        break

            print("\nTraining completed.")

            # ====================================================
            #            RESTORE BEST MODEL IN MEMORY
            # ====================================================
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                print("Restored best model weights into memory.")

            print(f"Best model saved to: {best_model_path}")
