import torch
from torch.utils.tensorboard import SummaryWriter


def train_mlp(model, train_loader, criterion, optimizer, device, epochs=1, writer=None):
    """
    Trains the MLP model for one epoch.

    Args:
        model (torch.nn.Module): The MLP model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device to run the training on (CPU or GPU).
        epochs (int): Number of epochs to train the model.
        writer (torch.utils.tensorboard.SummaryWriter, optional): TensorBoard writer for logging.

    Returns:
        float: Average loss over the training dataset.
    """
    model.train()
    total_loss = 0.0
    for epoch in range(epochs):
        step = 1
        for data, labels in train_loader:
            print(
                f"Epoch {epoch + 1}/{epochs}, Step {step}/{len(train_loader)}         ",
                end="\r",
            )
            step += 1
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if writer:
                writer.add_scalar(
                    "Loss/train", loss.item(), step + len(train_loader) * (epoch)
                )

    return total_loss / len(train_loader)


def evaluate_mlp(model, test_loader, criterion, device):
    """
    Evaluates the MLP model on the test dataset.

    Args:
        model (torch.nn.Module): The MLP model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test data.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the evaluation on (CPU or GPU).

    Returns:
        float: Average loss over the test dataset.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(test_loader)
