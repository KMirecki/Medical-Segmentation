from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class SegmentationDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        if len(img_paths) != len(mask_paths):
            raise ValueError("The number of images and masks must be equal")

        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        img = img.to(torch.float32)

        mask = (mask > 127).float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return img, mask


def get_paths(dataset_path: Path):
    img_dir = dataset_path / "images"
    mask_dir = dataset_path / "masks"

    img_paths = sorted(img_dir.glob("*.*"))
    mask_paths = sorted(mask_dir.glob("*.*"))

    return img_paths, mask_paths


def create_dataloaders(
    dataset_path: Path,
    train_transform,
    eval_transform,
    batch_size: int,
    num_workers: int = 4,
):
    img_paths, mask_paths = get_paths(dataset_path)

    X_train, X_temp, y_train, y_temp = train_test_split(
        img_paths, mask_paths, test_size=0.2, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    train_dataset = SegmentationDataset(X_train, y_train, transform=train_transform)
    val_dataset = SegmentationDataset(X_val, y_val, transform=eval_transform)
    test_dataset = SegmentationDataset(X_test, y_test, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
