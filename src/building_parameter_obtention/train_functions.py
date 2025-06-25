import torch
from torch.utils.tensorboard import SummaryWriter


def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for sequences, labels in train_loader:
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, labels)  # Unsqueeze labels to match output dim

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
