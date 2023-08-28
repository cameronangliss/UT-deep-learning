from PIL import Image
import csv
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv

        WARNING: Do not perform data normalization here. 
        """
        with open(os.path.join(dataset_path, "labels.csv")) as f:
            self.csv_dict = list(csv.DictReader(f))

    def __len__(self):
        """
        Your code here
        """
        return len(self.csv_dict)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        row = self.csv_dict[idx]
        return row["file"], row["label"]


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
