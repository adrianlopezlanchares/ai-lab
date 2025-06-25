import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, data, labels):
        """
        Initializes the dataset with data and labels.

        Args:
            data (torch.Tensor): The input data.
            labels (torch.Tensor): The corresponding labels.
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a sample and its label by index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the sample and its label.
        """
        return self.data[idx], self.labels[idx]
