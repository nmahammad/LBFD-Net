import torch
import torch.nn as nn

from lbfd_net.dataloader.binary_fall_detection_dataset import BinaryFallDataset
from lbfd_net.helpers.model_selection import choose_model_and_settings
from lbfd_net.helpers.get_computing_device import get_computing_device
from lbfd_net.helpers.constants import DATASET_DIRECTORY_PATH


MODEL_NAME: str = "lbfdnet"      # Options: "lbfdnet", "alexnet", "lenet"
SANITY_BATCH_SIZE: int = 16
SANITY_NUMBER_OF_EPOCHS: int = 50
SANITY_LEARNING_RATE: float = 1e-3
EARLY_STOPPING_PATIENCE: int = 10   


def run_sanity_check(model_name_string: str):
    """Run a one-batch overfitting sanity test with early stopping."""

    # ------------------------------------------------------------------
    # Step 1: Device
    # ------------------------------------------------------------------
    computing_device = get_computing_device()
    print("Using computing device:", computing_device)

    # ------------------------------------------------------------------
    # Step 2: Model + settings
    # ------------------------------------------------------------------
    (
        model_instance,
        normalization_type_string,
        expected_image_size_tuple,
    ) = choose_model_and_settings(model_name_string)

    print("\nSelected model:", model_name_string)
    print("Normalization:", normalization_type_string)
    print("Expected input size:", expected_image_size_tuple)

    model_instance = model_instance.to(computing_device)

    # ------------------------------------------------------------------
    # Step 3: Dataset
    # ------------------------------------------------------------------
    training_dataset = BinaryFallDataset(
        root_directory_path=DATASET_DIRECTORY_PATH,
        subset="train",
        use_augmentation=False,   # No augmentation for sanity test
        normalization_type=normalization_type_string,
        image_size=expected_image_size_tuple
    )

    training_dataset.load_dataset()

    training_data_loader = training_dataset.get_data_loader(
        batch_size=SANITY_BATCH_SIZE,
        shuffle_data=True
    )

    # ------------------------------------------------------------------
    # Step 4: Take exactly ONE batch
    # ------------------------------------------------------------------
    single_batch_images_tensor = None
    single_batch_labels_tensor = None

    for image_batch_tensor, label_batch_tensor in training_data_loader:
        single_batch_images_tensor = image_batch_tensor.to(computing_device)
        single_batch_labels_tensor = label_batch_tensor.float().unsqueeze(1).to(computing_device)
        break

    if single_batch_images_tensor is None:
        raise RuntimeError("ERROR: No batch retrieved from dataloader.")

    print("\nRetrieved batch tensors:")
    print("Images:", single_batch_images_tensor.shape)
    print("Labels:", single_batch_labels_tensor.shape)

    # ------------------------------------------------------------------
    # Step 5: Loss + optimizer
    # ------------------------------------------------------------------
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model_instance.parameters(),
        lr=SANITY_LEARNING_RATE
    )

    # ------------------------------------------------------------------
    # Step 6: Overfit on ONE batch + early stopping
    # ------------------------------------------------------------------
    print("\nStarting sanity overfitting with early stopping...\n")

    best_loss_value = float("inf")
    epochs_without_improvement = 0

    for epoch_index in range(1, SANITY_NUMBER_OF_EPOCHS + 1):

        model_instance.train()
        optimizer.zero_grad()

        model_output_tensor = model_instance(single_batch_images_tensor)
        loss_value = loss_function(model_output_tensor, single_batch_labels_tensor)

        loss_value.backward()
        optimizer.step()

        current_loss_value = loss_value.item()
        print(f"Epoch {epoch_index}/{SANITY_NUMBER_OF_EPOCHS}  - Loss: {current_loss_value:.6f}")

        # -------------------------------
        # Early Stopping Check
        # -------------------------------
        if current_loss_value < best_loss_value:
            best_loss_value = current_loss_value
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch_index} epochs.")
                break

    print("\nSanity check finished.")
    print("If the loss decreased strongly → Model + transforms + dataloader are correct.\n")


def main():
    run_sanity_check(model_name_string=MODEL_NAME)


if __name__ == "__main__":
    main()
