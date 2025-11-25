import torch
import torch.nn as nn
from lbfd_net.metrics.metrics_calculator import MetricsCalculator


def validate_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    metrics_calculator: MetricsCalculator,
    computing_device: torch.device
) -> tuple[float, float, float, float, float]:
    """
    Evaluate the model for one epoch on the validation dataset.

    Computes:
        - validation loss
        - accuracy
        - precision
        - recall
        - F1 score

    This function does NOT:
        - log anything
        - save any models
        - apply schedulers
    """

    model.eval()
    metrics_calculator.reset()
    total_validation_loss = 0.0

    with torch.no_grad():
        for images_tensor, labels_tensor in dataloader:

            images_tensor = images_tensor.to(computing_device)
            labels_tensor = labels_tensor.float().unsqueeze(1).to(computing_device)

            model_output = model(images_tensor)
            loss_value = loss_function(model_output, labels_tensor)

            total_validation_loss += loss_value.item() * images_tensor.size(0)

            metrics_calculator.update_predictions(
                logits=model_output,
                target_labels=labels_tensor
            )

    average_validation_loss = total_validation_loss / len(dataloader.dataset)

    return (
        average_validation_loss,
        metrics_calculator.compute_accuracy(),
        metrics_calculator.compute_precision(),
        metrics_calculator.compute_recall(),
        metrics_calculator.compute_f1_score(),
    )
