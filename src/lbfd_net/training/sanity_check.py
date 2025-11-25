import torch
import torch.nn as nn


from lbfd_net.dataloader.binary_fall_detection_dataset import BinaryFallDataset
from lbfd_net.model.light_weight_binary_fall_detection_network import LightWeightBinaryFallDetectionNetwork


DATASET_PATH = "dataset"  


def sanity_check():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = LightWeightBinaryFallDetectionNetwork().to(device)


    dataset = BinaryFallDataset(root_directory_path=DATASET_PATH, subset="train")
    dataset.load_dataset()
    dataloader = dataset.get_data_loader(batch_size=8)

    batch_images = None
    batch_labels = None

    for index, (images, labels) in enumerate(dataloader):
        if index == 0:  # pick first batch
            batch_images = images.to(device)
            batch_labels = labels.float().unsqueeze(1).to(device)
            break

    if batch_images is None:
        raise RuntimeError("Could not retrieve a batch. Check dataloader.")

    print("Sanity check batch shapes:", batch_images.shape, batch_labels.shape)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\nStarting sanity overfitting...")

    for epoch in range(50):
        model.train()

        optimizer.zero_grad()
        predictions = model(batch_images)
        loss = loss_function(predictions, batch_labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1:3d}/100 - Loss: {loss.item():.6f}")

    print("\nSanity check completed. If loss dropped significantly → network works.")


if __name__ == "__main__":
    sanity_check()
