import torch
from torch.utils.data import Dataset


class MLPDataset(Dataset):
    def __init__(self, windows):
        """
        Initializes the dataset with data and labels.

        Args:
            data (torch.Tensor): The input data.
            labels (torch.Tensor): The corresponding labels.
        """
        self.data = torch.tensor(windows[0], dtype=torch.float32)
        self.labels = torch.tensor(windows[1], dtype=torch.float32)

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
