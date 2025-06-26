import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


def train(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    writer: SummaryWriter = None,
    verbose: bool = False,
):
    """Trains the model using the provided training data.

    Args:
        model (nn.Module): The pytorch model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset, providing batches of data.
        criterion (nn.Module): Loss function to compute the error between predictions and true labels.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters based on the computed gradients.
        num_epochs (int): Number of epochs to train the model.
        device (torch.device, optional): Device to run the training on (CPU or GPU).
        writer (SummaryWriter, optional): TensorBoard writer for logging training metrics. Defaults to None.
        verbose (bool, optional): If True, prints training progress. Defaults to False.
    """
    model.train()
    step = 0
    for epoch in range(num_epochs):
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if writer:
                writer.add_scalar("Loss/train", loss.item(), epoch)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Step {step + 1}/{len(train_loader)}       ",
                    end="\r",
                )


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = "cpu",
):
    """Evaluates the model on the test dataset.

    Args:
        model (nn.Module): The pytorch model to be evaluated.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (nn.Module): Loss function to compute the error between predictions and true labels.
        device (torch.device, optional): Device to run the evaluation on (CPU or GPU). Defaults to "cpu".

    Returns:
        tuple: A tuple containing:
            - predictions (list): List of predictions made by the model.
            - average_loss (float): Average loss over the test set.
    """
    model.eval()
    total_loss = 0.0
    predictions = []
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences).squeeze()
            predictions.extend(outputs.cpu().numpy().tolist())
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return predictions, total_loss / len(test_loader)  # Average loss over the test set
