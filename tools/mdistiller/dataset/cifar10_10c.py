import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image


def get_data_folder():
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "/home/dnh/datasets")
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    return data_folder


class CIFAR10_10CInstance(Dataset):
    """Dataset combining CIFAR-10 and CIFAR-10-C"""

    def __init__(self, cifar10_data, cifar10_labels, cifar10c_data, cifar10c_labels, transform=None):
        self.cifar10_data = cifar10_data
        self.cifar10_labels = cifar10_labels
        self.cifar10c_data = cifar10c_data
        self.cifar10c_labels = cifar10c_labels
        self.transform = transform
        self.length = len(cifar10_data) + len(cifar10c_data)

    def __getitem__(self, index):
        if index < len(self.cifar10_data):
            img, target = self.cifar10_data[index], self.cifar10_labels[index]
        else:
            img, target = self.cifar10c_data[index - len(self.cifar10_data)], self.cifar10c_labels[
                index - len(self.cifar10_data)]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        # 返回 img, target, index
        return img, target, index

    def __len__(self):
        return self.length


def get_cifar10_10c_dataloaders(batch_size, val_batch_size, num_workers, corruptions):
    data_folder = get_data_folder()

    # Load CIFAR-10
    train_transform = get_cifar10_train_transform()
    test_transform = get_cifar10_test_transform()

    cifar10_train_set = datasets.CIFAR10(
        root=data_folder, download=True, train=True, transform=train_transform
    )
    cifar10_data, cifar10_labels = cifar10_train_set.data, cifar10_train_set.targets

    # Load CIFAR-10-C with specified corruptions
    cifar10c_data = []
    cifar10c_labels = []
    cifar10c_data_folder = os.path.join(data_folder, "CIFAR-10-C")

    for corruption in corruptions:
        corruption_file = os.path.join(cifar10c_data_folder, f"{corruption}.npy")
        if not os.path.isfile(corruption_file):
            raise FileNotFoundError(f"{corruption_file} not found. Please ensure the CIFAR-10-C dataset is available.")

        corruption_data = np.load(corruption_file)
        corruption_labels = np.load(os.path.join(cifar10c_data_folder, "labels.npy"))

        cifar10c_data.append(corruption_data)
        cifar10c_labels.append(corruption_labels)

    # Concatenate all corruption data and labels
    cifar10c_data = np.concatenate(cifar10c_data, axis=0)
    cifar10c_labels = np.concatenate(cifar10c_labels, axis=0)

    # Create combined dataset
    combined_dataset = CIFAR10_10CInstance(
        cifar10_data, cifar10_labels, cifar10c_data, cifar10c_labels, transform=train_transform
    )

    num_data = len(combined_dataset)
    train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Use the CIFAR-10 test set for validation
    cifar10_test_set = datasets.CIFAR10(
        root=data_folder, download=True, train=False, transform=test_transform
    )
    val_loader = DataLoader(
        cifar10_test_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers
    )

    num_classes = 10  # CIFAR-10 and CIFAR-10-C have 10 classes

    return train_loader, val_loader, num_data, num_classes


def get_cifar10_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


def get_cifar10_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
