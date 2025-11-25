import torch
import torch.nn as nn


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    computing_device: torch.device
) -> float:
    """
    Train the model for a single epoch and return the average training loss.
    """

    model.train()
    total_training_loss = 0.0

    for images_tensor, labels_tensor in dataloader:

        images_tensor = images_tensor.to(computing_device)
        labels_tensor = labels_tensor.float().unsqueeze(1).to(computing_device)

        optimizer.zero_grad()
        model_output = model(images_tensor)
        loss_value = loss_function(model_output, labels_tensor)
        loss_value.backward()
        optimizer.step()

        # accumulate weighted loss (batch_size)
        total_training_loss += loss_value.item() * images_tensor.size(0)

    average_training_loss = total_training_loss / len(dataloader.dataset)
    return average_training_loss
