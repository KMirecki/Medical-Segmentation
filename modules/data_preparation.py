import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


class SegmentationDataset(Dataset):
    def __init__(self, img_paths: list, mask_paths: list, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

        if len(self.img_paths) != len(self.mask_paths):
            raise ValueError("The length of img_paths and mask_paths do not match")

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

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        mask = mask.to(torch.float32)
        mask = (mask > 127).to(torch.float32)

        return img, mask


def get_paths(dataset_path: Path):
    img_dir = dataset_path / "images"
    mask_dir = dataset_path / "masks"

    img_paths = sorted(list(img_dir.glob("*.*")))
    mask_paths = sorted(list(mask_dir.glob("*.*")))
    return img_paths, mask_paths


def create_dataloaders(dataset_path: Path,
                       train_transform,
                       test_transform,
                       batch_size: int):
    img_paths, mask_paths = get_paths(dataset_path)

    X_train, X_temp, y_train, y_temp = train_test_split(
        img_paths, mask_paths, test_size=0.2, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    train_dataset = SegmentationDataset(X_train, y_train, transform=train_transform)
    val_dataset = SegmentationDataset(X_val, y_val, transform=test_transform)
    test_dataset = SegmentationDataset(X_test, y_test, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
