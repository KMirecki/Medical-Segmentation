import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch


def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
):
    all_tp, all_fp, all_fn, all_tn = [], [], [], []

    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            probs = torch.sigmoid(logits)

            tp, fp, fn, tn = smp.metrics.get_stats(
                probs, y.long(), mode="binary", threshold=0.5
            )
            all_tp.append(tp.cpu())
            all_fp.append(fp.cpu())
            all_fn.append(fn.cpu())
            all_tn.append(tn.cpu())

        all_tp = torch.cat(all_tp)
        all_fp = torch.cat(all_fp)
        all_fn = torch.cat(all_fn)
        all_tn = torch.cat(all_tn)

        metric_iou = smp.metrics.iou_score(
            all_tp, all_fp, all_fn, all_tn, reduction="micro"
        )
        metric_f1 = smp.metrics.f1_score(
            all_tp, all_fp, all_fn, all_tn, reduction="micro"
        )
        metric_precision = smp.metrics.precision(
            all_tp, all_fp, all_fn, all_tn, reduction="micro"
        )
        metric_recall = smp.metrics.recall(
            all_tp, all_fp, all_fn, all_tn, reduction="micro"
        )

    return {
        "test_iou": metric_iou.item(),
        "test_f1": metric_f1.item(),
        "test_precision": metric_precision.item(),
        "test_recall": metric_recall.item(),
    }


def visualize_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n=5,
):
    model.eval()

    imgs, masks = next(iter(dataloader))
    batch_size = imgs.shape[0]

    n = min(n, batch_size)
    random_samples_idx = random.sample(range(batch_size), n)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(15, n * 4))  # noqa: RUF059
    for col, title in enumerate(["Image", "Ground Truth", "Prediction"]):
        axes[0, col].set_title(title, fontweight="bold")

    with torch.inference_mode():
        for i, targ_sample in enumerate(random_samples_idx):
            img = imgs[targ_sample]
            mask = masks[targ_sample]

            logit = model(img.unsqueeze(0).to(device))
            pred_np = (torch.sigmoid(logit) > 0.5).squeeze().cpu().numpy()

            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np * std + mean, 0, 1)

            mask_np = mask.squeeze().cpu().numpy()

            ax_img = axes[i, 0]
            ax_img.imshow(img_np)
            ax_img.axis("off")

            ax_mask = axes[i, 1]
            ax_mask.imshow(mask_np, cmap="gray")
            ax_mask.axis("off")

            ax_mask = axes[i, 2]
            ax_mask.imshow(pred_np, cmap="gray")
            ax_mask.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_single_prediction(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.eval()

    imgs, masks = next(iter(dataloader))
    batch_size = imgs.shape[0]
    targ_sample = random.randint(0, batch_size - 1)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(22, 5))  # noqa: RUF059
    titles = [
        "Image",
        "Ground Truth",
        "Prediction",
        "Errors (FP:Red, FN:Blue, TP:Green)",
    ]

    for col, title in enumerate(titles):
        axes[col].set_title(title, fontweight="bold")

    with torch.inference_mode():
        img = imgs[targ_sample]
        mask = masks[targ_sample]

        logit = model(img.unsqueeze(0).to(device))
        pred_np = (torch.sigmoid(logit) > 0.5).squeeze().cpu().numpy().astype(bool)

        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * std + mean, 0, 1)

        mask_np = mask.squeeze().cpu().numpy().astype(bool)

        h, w = mask_np.shape
        error_map = np.zeros((h, w, 3), dtype=np.float32)

        error_map[pred_np & ~mask_np] = [1.0, 0.0, 0.0]
        error_map[~pred_np & mask_np] = [0.0, 0.0, 1.0]
        error_map[pred_np & mask_np] = [0.0, 1.0, 0.0]

        axes[0].imshow(img_np)
        axes[1].imshow(mask_np, cmap="gray")
        axes[2].imshow(pred_np, cmap="gray")
        axes[3].imshow(error_map)

        for ax in axes:
            ax.axis("off")

    plt.tight_layout(h_pad=1.0, w_pad=2.5)
    plt.show()


def compare_models_predictions(
    models_with_names: List[Tuple[str, torch.nn.Module]],  # noqa: UP006
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    seed: int = 42,
):
    num_models = len(models_with_names)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    imgs, masks = next(iter(dataloader))
    batch_size = imgs.shape[0]
    targ_sample = random.randint(0, batch_size - 1)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = imgs[targ_sample]
    mask = masks[targ_sample]

    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np * std + mean, 0, 1)
    mask_np = mask.squeeze().cpu().numpy().astype(bool)

    fig, axes = plt.subplots(nrows=num_models, ncols=4, figsize=(22, num_models * 4.5))  # noqa: RUF059

    if num_models == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = [
        "Image",
        "Ground Truth",
        "Prediction",
        "Errors (FP:Red, FN:Blue, TP:Green)",
    ]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontweight="bold", fontsize=14, pad=15)

    for row_idx, (model_name, model) in enumerate(models_with_names):
        model.eval()

        with torch.inference_mode():
            logit = model(img.unsqueeze(0).to(device))
            pred_np = (torch.sigmoid(logit) > 0.5).squeeze().cpu().numpy().astype(bool)

            h, w = mask_np.shape
            error_map = np.zeros((h, w, 3), dtype=np.float32)
            error_map[pred_np & ~mask_np] = [1.0, 0.0, 0.0]
            error_map[~pred_np & mask_np] = [0.0, 0.0, 1.0]
            error_map[pred_np & mask_np] = [0.0, 1.0, 0.0]

            axes[row_idx, 0].imshow(img_np)
            axes[row_idx, 1].imshow(mask_np, cmap="gray")
            axes[row_idx, 2].imshow(pred_np, cmap="gray")
            axes[row_idx, 3].imshow(error_map)

            axes[row_idx, 0].text(
                -0.15,
                0.5,
                model_name,
                transform=axes[row_idx, 0].transAxes,
                fontweight="bold",
                fontsize=14,
                ha="right",
                va="center",
            )

            for col_idx in range(4):
                axes[row_idx, col_idx].axis("off")

    plt.tight_layout(h_pad=2.0, w_pad=2.0)
    plt.show()
