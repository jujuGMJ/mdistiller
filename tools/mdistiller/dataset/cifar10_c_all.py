import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

class CIFAR10CInstance:
    def __init__(self, data, labels, severity, transform=None):
        self.data = data
        self.targets = labels
        self.severity = severity
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, self.severity[index]

    def __len__(self):
        return len(self.data)

def get_cifar10_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

def calculate_severity(data, num_severity_levels=5):
    num_samples_per_severity = len(data) // num_severity_levels
    severity = np.concatenate([
        np.full(num_samples_per_severity, i+1) for i in range(num_severity_levels)
    ])
    return severity

def get_cifar10c_all_dataloader(corruption, batch_size, num_workers):
    data_folder = "/home/dnh/datasets/CIFAR-10-C"
    corruption_file = os.path.join(data_folder, f"{corruption}.npy")
    labels_file = os.path.join(data_folder, "labels.npy")

    if not os.path.isfile(corruption_file):
        raise FileNotFoundError(f"{corruption_file} not found.")
    if not os.path.isfile(labels_file):
        raise FileNotFoundError(f"{labels_file} not found.")

    data = np.load(corruption_file)
    labels = np.load(labels_file)
    severity = calculate_severity(data)

    transform = get_cifar10_test_transform()
    dataset = CIFAR10CInstance(data, labels, severity, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloader
