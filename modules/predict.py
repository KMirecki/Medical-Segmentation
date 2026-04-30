import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import random
import numpy as np


def predict(model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            device: torch.device):
    all_tp, all_fp, all_fn, all_tn = [], [], [], []

    model.eval()
    with torch.inference_mode():
        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            probs = torch.sigmoid(logits)

            tp, fp, fn, tn = smp.metrics.get_stats(probs, y.long(), mode='binary', threshold=0.5)
            all_tp.append(tp)
            all_fp.append(fp)
            all_fn.append(fn)
            all_tn.append(tn)

        all_tp = torch.cat(all_tp)
        all_fp = torch.cat(all_fp)
        all_fn = torch.cat(all_fn)
        all_tn = torch.cat(all_tn)

        metric_iou = smp.metrics.iou_score(all_tp, all_fp, all_fn, all_tn, reduction="micro")
        metric_f1 = smp.metrics.f1_score(all_tp, all_fp, all_fn, all_tn, reduction="micro")
        metric_precision = smp.metrics.precision(all_tp, all_fp, all_fn, all_tn, reduction="micro")
        metric_recall = smp.metrics.recall(all_tp, all_fp, all_fn, all_tn, reduction="micro")

    return {"test_iou": metric_iou.item(),
            "test_f1": metric_f1.item(),
            "test_precision": metric_precision.item(),
            "test_recall": metric_recall.item()}


def visualize_predictions(model: torch.nn.Module,
                          dataloader: torch.utils.data.DataLoader,
                          device: torch.device,
                          n=5):
    model.eval()

    imgs, masks = next(iter(dataloader))
    batch_size = imgs.shape[0]
    random_samples_idx = random.sample(range(batch_size), n)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(15, n * 4))
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
            ax_img.axis('off')

            ax_mask = axes[i, 1]
            ax_mask.imshow(mask_np, cmap='gray')
            ax_mask.axis('off')

            ax_mask = axes[i, 2]
            ax_mask.imshow(pred_np, cmap='gray')
            ax_mask.axis('off')

    plt.tight_layout()
    plt.show()
