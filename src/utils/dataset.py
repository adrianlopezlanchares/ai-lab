import numpy as np
import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
    """Dataset for building parameter prediction.

    Attributes:
        features (torch.Tensor): Tensor containing the input features.
        labels (torch.Tensor): Tensor containing the target labels.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """Initializes the dataset with features and labels.

        Args:
            features (np.ndarray): Input features.
            labels (np.ndarray): Target labels.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx):
        """Returns a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the features and labels for the sample.
        """
        return self.features[idx], self.labels[idx]
