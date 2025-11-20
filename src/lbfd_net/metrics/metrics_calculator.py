import numpy as np
import torch
from typing import List, Union
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)



class MetricsCalculator:
    """
    Computes and stores binary classification metrics for fall detection.
    Tracks predictions and ground truth labels over an entire epoch.
    """

    def __init__(self) -> None:
        """Initialize empty lists for predictions and ground truth labels."""
        self.ground_truth_labels_list: List[int] = []
        self.model_predicted_labels_list: List[int] = []
    
    @staticmethod
    def sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
        """Apply the sigmoid function to raw logits."""
        return 1.0 / (1.0 + np.exp(-logits))


    def update_predictions(
        self,
        logits: Union[np.ndarray, torch.Tensor],
        target_labels: Union[np.ndarray, torch.Tensor],
        threshold: float = 0.5
    ) -> None:
        """
        Convert raw logits into binary predictions using sigmoid + threshold,
        and store predictions and ground truth labels.

        Args:
            logits: Model outputs of shape [batch_size] or [batch_size, 1]
            target_labels: Ground truth labels of shape [batch_size]
            threshold: Decision threshold for classification (default 0.5)
        """

        # Convert PyTorch tensors to NumPy
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()

        if isinstance(target_labels, torch.Tensor):
            target_labels = target_labels.detach().cpu().numpy()

        # Ensure logits are 2D
        logits = logits.reshape(-1, 1)
        target_labels = target_labels.reshape(-1)

        # Check shape consistency
        if logits.shape[0] != target_labels.shape[0]:
            raise ValueError(
                f"Mismatch between logits batch size ({logits.shape[0]}) and target size ({target_labels.shape[0]})"
            )

        # Convert logits → probabilities
        probabilities = self.sigmoid_numpy(logits)

        # Apply threshold
        binary_predictions = (probabilities >= threshold).astype(int).flatten()

        # Store values
        self.ground_truth_labels_list.extend(target_labels.tolist())
        self.model_predicted_labels_list.extend(binary_predictions.tolist())


    # ----------------------
    # Metric computations
    # ----------------------

    def compute_precision(self) -> float:
        return precision_score(
            self.ground_truth_labels_list,
            self.model_predicted_labels_list,
            zero_division=0
        )

    def compute_recall(self) -> float:
        return recall_score(
            self.ground_truth_labels_list,
            self.model_predicted_labels_list,
            zero_division=0
        )

    def compute_f1_score(self) -> float:
        return f1_score(
            self.ground_truth_labels_list,
            self.model_predicted_labels_list,
            zero_division=0
        )

    def compute_accuracy(self) -> float:
        return accuracy_score(
            self.ground_truth_labels_list,
            self.model_predicted_labels_list
        )

    def compute_confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(
            self.ground_truth_labels_list,
            self.model_predicted_labels_list
        )

    def reset(self) -> None:
        """Clear stored predictions and labels."""
        self.ground_truth_labels_list = []
        self.model_predicted_labels_list = []


