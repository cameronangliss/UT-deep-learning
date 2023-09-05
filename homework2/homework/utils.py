import csv
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        with open(os.path.join(dataset_path, "labels.csv")) as f:
            self.data = list(csv.DictReader(f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        image_to_tensor = transforms.ToTensor()
        image_path = os.path.join(self.dataset_path, row["file"])
        image_tensor = image_to_tensor(Image.open(image_path))
        label_id = LABEL_NAMES.index(row["label"])
        return image_tensor, label_id


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
