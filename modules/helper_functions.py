import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'")


def dataloaders_info(train_loader,
                     val_loader,
                     test_loader,
                     batch_size):
    print(f"the size of the train dataloader : {len(train_loader)} batches of {batch_size}")
    print(f"the size of the validation dataloader : {len(val_loader)} batches of {batch_size}")
    print(f"the size of the test dataloader : {len(test_loader)} batches of {batch_size}")


def display_random_images(dataloader: torch.utils.data.DataLoader,
                          n: int = 5):
    imgs, masks = next(iter(dataloader))
    batch_size = imgs.shape[0]

    random_samples_idx = random.sample(range(batch_size), n)

    fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(n * 3, 6))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i, targ_sample in enumerate(random_samples_idx):
        img = imgs[targ_sample]
        mask = masks[targ_sample]

        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * std + mean, 0, 1)

        mask_np = mask.squeeze().cpu().numpy()

        ax_img = axes[0, i] if n > 1 else axes[0]
        ax_img.imshow(img_np)
        ax_img.set_title(f"Batch idx: {targ_sample}")
        ax_img.axis('off')

        ax_mask = axes[1, i] if n > 1 else axes[1]
        ax_mask.imshow(mask_np, cmap='gray')
        ax_mask.set_title("Mask")
        ax_mask.axis('off')

    plt.tight_layout()
    plt.show()


def plot_loss_curves(results, title):
    train_loss = results['train_loss']
    train_iou = results['train_iou']
    train_f1 = results['train_f1']
    val_loss = results['val_loss']
    val_iou = results['val_iou']
    val_f1 = results['val_f1']
    epochs = range(len(results["train_loss"]))
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Val loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_iou, label="Train IoU")
    plt.plot(epochs, val_iou, label="Val IoU")
    plt.title("IoU")
    plt.xlabel("Epochs")
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_f1, label="Train F1")
    plt.plot(epochs, val_f1, label="Val F1")
    plt.title("F1 Score")
    plt.xlabel("Epochs")
    plt.legend()
    plt.suptitle(title)
    plt.show()


def create_writer(dataset_name: str,
                  model_name: str,
                  extra: str = None):
    if extra:
        log_dir = os.path.join("runs", dataset_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", dataset_name, model_name)

    return SummaryWriter(log_dir=log_dir)
