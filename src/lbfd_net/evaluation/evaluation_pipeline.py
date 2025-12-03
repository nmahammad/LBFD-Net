from pathlib import Path
import torch
import torch.nn as nn
import mlflow

from lbfd_net.dataloader.binary_fall_detection_dataset import BinaryFallDataset
from lbfd_net.metrics.metrics_calculator import MetricsCalculator
from lbfd_net.helpers.model_selection import choose_model_and_settings
from lbfd_net.helpers.get_computing_device import get_computing_device
from lbfd_net.helpers.constants import DATASET_DIRECTORY_PATH, MODEL_WEIGHTS_ROOT_DIRECTORY


class EvaluationPipeline:
    """
    A complete evaluation system which:

    - Finds the correct trained model run directory
    - Loads trained weights into the correct model architecture
    - Builds the test dataset and DataLoader
    - Computes evaluation metrics (loss, accuracy, precision, recall, F1 score)
    - Generates a confusion matrix
    - Saves output files
    - Optionally logs everything to MLflow
    """

    def __init__(
        self,
        model_name: str,
        test_batch_size: int,
        manual_run_index: int | None = None,
        log_to_mlflow: bool = False,
        mlflow_experiment_name: str = "evaluation_experiment"
    ):
        self.model_name = model_name
        self.test_batch_size = test_batch_size
        self.manual_run_index = manual_run_index
        self.log_to_mlflow = log_to_mlflow
        self.mlflow_experiment_name = mlflow_experiment_name


    def find_all_model_runs(self) -> list[Path]:
        """
        Find all run directories that belong to this model.
        """

        model_weights_root = Path(MODEL_WEIGHTS_ROOT_DIRECTORY)

        if not model_weights_root.exists():
            raise FileNotFoundError(
                f"Model weights root directory does not exist: {MODEL_WEIGHTS_ROOT_DIRECTORY}"
            )

        discovered_run_directories: list[Path] = []

        for directory_entry in model_weights_root.iterdir():
            directory_name = directory_entry.name

            if directory_entry.is_dir() and directory_name.startswith("run_"):

                # Structure example: run_1_lbfdnet
                parts = directory_name.split("_")
                if len(parts) >= 3:
                    directory_model_name = parts[2]

                    if directory_model_name == self.model_name:
                        discovered_run_directories.append(directory_entry)

        return discovered_run_directories

    def get_latest_run_for_model(self) -> Path:
        """
        Return the most recently modified run directory for the selected model name.
        """

        list_of_matching_run_directories = self.find_all_model_runs()

        if len(list_of_matching_run_directories) == 0:
            raise FileNotFoundError(
                f"No run directories found for model name '{self.model_name}'."
            )

        # Sort by modification time (oldest → newest)
        list_of_matching_run_directories.sort(
            key=lambda directory_path: directory_path.stat().st_mtime
        )

        latest_run_directory_path = list_of_matching_run_directories[-1]
        return latest_run_directory_path

    def get_run_directory_by_index(self, run_index: int) -> Path:
        """
        Retrieve a run folder by index explicitly:
        Example: run_5_lbfdnet
        """
        run_directory_path = Path(MODEL_WEIGHTS_ROOT_DIRECTORY) / f"run_{run_index}_{self.model_name}"

        if not run_directory_path.exists():
            raise FileNotFoundError(
                f"Run directory with index {run_index} does not exist for model {self.model_name}"
            )

        return run_directory_path
    
    def load_model_with_weights(self, model_weights_file_path: Path):
        """
        Load model architecture + trained weights.
        """

        (
            model_instance,
            normalization_type_string,
            expected_image_size_tuple,
        ) = choose_model_and_settings(self.model_name)

        computing_device = get_computing_device()

        loaded_state_dictionary = torch.load(
            model_weights_file_path, map_location=computing_device
        )

        model_instance.load_state_dict(loaded_state_dictionary)
        model_instance.to(computing_device)

        return (
            model_instance,
            normalization_type_string,
            expected_image_size_tuple,
            computing_device,
        )

    def create_test_data_loader(
        self,
        normalization_type_string: str,
        expected_image_size_tuple: tuple[int, int],
    ):
        test_dataset_instance = BinaryFallDataset(
            root_directory_path=DATASET_DIRECTORY_PATH,
            subset="test",
            use_augmentation=False,
            normalization_type=normalization_type_string,
            image_size=expected_image_size_tuple
        )

        test_dataset_instance.load_dataset()

        test_data_loader_instance = test_dataset_instance.get_data_loader(
            batch_size=self.test_batch_size,
            shuffle_data=False
        )

        return test_data_loader_instance


    def evaluate(self):
        # --------------------------------------------------------------
        # Step 1 — Locate the correct run directory
        # --------------------------------------------------------------
        if self.manual_run_index is None:
            selected_run_directory_path = self.get_latest_run_for_model()
            print("Selected latest run directory:", selected_run_directory_path)
        else:
            selected_run_directory_path = self.get_run_directory_by_index(
                self.manual_run_index
            )
            print("Selected manual run directory:", selected_run_directory_path)

        # --------------------------------------------------------------
        # Step 2 — Load model and weights
        # --------------------------------------------------------------
        best_model_weights_path = selected_run_directory_path / "best_model.pth"
        print("Loading model weights from:", best_model_weights_path)

        (
            model_instance,
            normalization_type_string,
            expected_image_size_tuple,
            computing_device,
        ) = self.load_model_with_weights(best_model_weights_path)

        # --------------------------------------------------------------
        # Step 3 — Build test dataset + dataloader
        # --------------------------------------------------------------
        test_data_loader_instance = self.create_test_data_loader(
            normalization_type_string,
            expected_image_size_tuple
        )

        # --------------------------------------------------------------
        # Step 4 — Evaluation loop
        # --------------------------------------------------------------
        model_instance.eval()
        metrics_calculator = MetricsCalculator()
        loss_function = nn.BCEWithLogitsLoss()

        accumulated_loss_value = 0.0

        # MLflow setup (optional)
        if self.log_to_mlflow:
            mlflow.set_tracking_uri("mlruns")
            mlflow.set_experiment(self.mlflow_experiment_name)
            active_mlflow_run = mlflow.start_run(
                run_name=f"evaluation_{self.model_name}"
            )
        else:
            active_mlflow_run = None

        with torch.no_grad():
            for image_batch_tensor, label_batch_tensor in test_data_loader_instance:

                image_batch_tensor = image_batch_tensor.to(computing_device)
                label_batch_tensor = label_batch_tensor.to(computing_device)

                if label_batch_tensor.ndim == 1:
                    label_batch_tensor = label_batch_tensor.float().unsqueeze(1)

                prediction_tensor = model_instance(image_batch_tensor)
                batch_loss_value = loss_function(prediction_tensor, label_batch_tensor)

                accumulated_loss_value += batch_loss_value.item() * image_batch_tensor.size(0)

                metrics_calculator.update_predictions(
                    logits=prediction_tensor,
                    target_labels=label_batch_tensor
                )

        average_loss_value = accumulated_loss_value / len(test_data_loader_instance.dataset)

        evaluation_results = {
            "loss": average_loss_value,
            "accuracy": metrics_calculator.compute_accuracy(),
            "precision": metrics_calculator.compute_precision(),
            "recall": metrics_calculator.compute_recall(),
            "f1_score": metrics_calculator.compute_f1_score(),
            "confusion_matrix": metrics_calculator.compute_confusion_matrix(),
        }

        # --------------------------------------------------------------
        # Step 5 — MLflow logging
        # --------------------------------------------------------------
        if active_mlflow_run is not None:
            mlflow.log_metric("test_loss", evaluation_results["loss"])
            mlflow.log_metric("test_accuracy", evaluation_results["accuracy"])
            mlflow.log_metric("test_precision", evaluation_results["precision"])
            mlflow.log_metric("test_recall", evaluation_results["recall"])
            mlflow.log_metric("test_f1_score", evaluation_results["f1_score"])
            mlflow.end_run()

        # --------------------------------------------------------------
        # Step 6 — Save confusion matrix to disk
        # --------------------------------------------------------------
        confusion_matrix_output_path = selected_run_directory_path / "confusion_matrix.txt"

        with open(confusion_matrix_output_path, "w") as file_handle:
            file_handle.write(str(evaluation_results["confusion_matrix"]))

        print("Confusion matrix saved to:", confusion_matrix_output_path)

        # --------------------------------------------------------------
        # Step 7 — Pretty print results
        # --------------------------------------------------------------
        print("\n===== FINAL EVALUATION RESULTS =====")
        print(f"Loss       : {evaluation_results['loss']:.4f}")
        print(f"Accuracy   : {evaluation_results['accuracy']:.4f}")
        print(f"Precision  : {evaluation_results['precision']:.4f}")
        print(f"Recall     : {evaluation_results['recall']:.4f}")
        print(f"F1 Score   : {evaluation_results['f1_score']:.4f}")
        print("\nConfusion Matrix:")
        print(evaluation_results["confusion_matrix"])

        return evaluation_results
