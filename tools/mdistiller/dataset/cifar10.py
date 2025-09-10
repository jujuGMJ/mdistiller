import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from sklearn.model_selection import train_test_split

def get_data_folder():
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "/home/gmj/datasets")
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    return data_folder

class CIFAR10Instance(datasets.CIFAR10):
    """CIFAR10Instance Dataset."""

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index

class CIFAR10InstanceSample(datasets.CIFAR10):
    """CIFAR10Instance+Sample Dataset"""

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, k=4096, mode="exact", is_sample=True, percent=1.0):
        super().__init__(root=root, train=train, download=download, transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 10
        num_samples = len(self.data)
        label = self.targets

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n] for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            return img, target, index
        else:
            if self.mode == "exact":
                pos_idx = index
            elif self.mode == "relax":
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx

def get_cifar10_train_transform():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    return train_transform

def get_cifar10_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

def get_cifar10_dataloaders(batch_size, val_batch_size, num_workers):
    data_folder = get_data_folder()
    train_transform = get_cifar10_train_transform()
    test_transform = get_cifar10_test_transform()
    train_set = CIFAR10Instance(root=data_folder, download=True, train=True, transform=train_transform)
    num_data = len(train_set)
    test_set = datasets.CIFAR10(root=data_folder, download=True, train=False, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=1)
    return train_loader, test_loader, num_data

# CIFAR-10 for CRD
def get_cifar10_dataloaders_sample(batch_size, val_batch_size, num_workers, k, mode="exact"):
    data_folder = get_data_folder()
    train_transform = get_cifar10_train_transform()
    test_transform = get_cifar10_test_transform()

    train_set = CIFAR10InstanceSample(root=data_folder, download=True, train=True, transform=train_transform, k=k, mode=mode, is_sample=True, percent=1.0)
    num_data = len(train_set)
    test_set = datasets.CIFAR10(root=data_folder, download=True, train=False, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, num_data

class CIFAR10CInstance:
    """A class specifically for loading CIFAR-10-C dataset"""
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

def calculate_severity(data, num_severity_levels=5):
    """根据数据索引推导腐蚀严重程度"""
    num_samples_per_severity = len(data) // num_severity_levels
    severity = np.concatenate([
        np.full(num_samples_per_severity, i+1) for i in range(num_severity_levels)
    ])
    return severity

def split_dataset_with_severity(data, labels, severity, train_ratio=0.7, random_state=42):
    combined_labels = np.array([f"{l}_{s}" for l, s in zip(labels, severity)])
    train_data, val_data, train_labels, val_labels, train_sev, val_sev = train_test_split(
        data, labels, severity, train_size=train_ratio, stratify=combined_labels, random_state=random_state
    )
    return (train_data, train_labels, train_sev), (val_data, val_labels, val_sev)

def get_cifar10c_dataloaders(corruption, batch_size, num_workers, train_ratio=0.7):
    data_folder = "/home/gmj/datasets/CIFAR-10-C"
    corruption_file = os.path.join(data_folder, f"{corruption}.npy")

    if not os.path.isfile(corruption_file):
        raise FileNotFoundError(f"{corruption_file} not found. Please ensure the CIFAR-10-C dataset is available.")

    data = np.load(corruption_file)
    labels_file = os.path.join(data_folder, "labels.npy")
    labels = np.load(labels_file)

    # Check if labels are in range
    print(f"Labels range: min={labels.min()}, max={labels.max()}")

    severity = calculate_severity(data)

    train_set, val_set = split_dataset_with_severity(data, labels, severity, train_ratio=train_ratio)

    transform = get_cifar10_test_transform()
    train_dataset = CIFAR10CInstance(train_set[0], train_set[1], train_set[2], transform=transform)
    val_dataset = CIFAR10CInstance(val_set[0], val_set[1], val_set[2], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

