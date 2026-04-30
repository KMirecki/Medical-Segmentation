import torch
from tqdm import tqdm
import segmentation_models_pytorch as smp
import time


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    train_loss = 0
    all_tp, all_fp, all_fn, all_tn = [], [], [], []

    model.train()

    for (X, y) in dataloader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        train_loss += loss.item()

        probs = torch.sigmoid(logits)

        tp, fp, fn, tn = smp.metrics.get_stats(probs, y.long(), mode='binary', threshold=0.5)
        all_tp.append(tp)
        all_fp.append(fp)
        all_fn.append(fn)
        all_tn.append(tn)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)

    all_tp = torch.cat(all_tp)
    all_fp = torch.cat(all_fp)
    all_fn = torch.cat(all_fn)
    all_tn = torch.cat(all_tn)

    metric_iou = smp.metrics.iou_score(all_tp, all_fp, all_fn, all_tn, reduction="micro")
    metric_f1 = smp.metrics.f1_score(all_tp, all_fp, all_fn, all_tn, reduction="micro")

    return train_loss, metric_iou.item(), metric_f1.item()


def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn,
             device: torch.device):
    val_loss = 0
    all_tp, all_fp, all_fn, all_tn = [], [], [], []

    model.eval()
    with torch.inference_mode():
        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            val_loss += loss.item()

            probs = torch.sigmoid(logits)

            tp, fp, fn, tn = smp.metrics.get_stats(probs, y.long(), mode='binary', threshold=0.5)
            all_tp.append(tp)
            all_fp.append(fp)
            all_fn.append(fn)
            all_tn.append(tn)

    val_loss /= len(dataloader)

    all_tp = torch.cat(all_tp)
    all_fp = torch.cat(all_fp)
    all_fn = torch.cat(all_fn)
    all_tn = torch.cat(all_tn)

    metric_iou = smp.metrics.iou_score(all_tp, all_fp, all_fn, all_tn, reduction="micro")
    metric_f1 = smp.metrics.f1_score(all_tp, all_fp, all_fn, all_tn, reduction="micro")

    return val_loss, metric_iou.item(), metric_f1.item()


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          loss_fn,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter,
          save_path: str,
          patience: int = 5):
    results = {"train_loss": [],
               "train_iou": [],
               "train_f1": [],
               "val_loss": [],
               "val_iou": [],
               "val_f1": [],
               "train_time_per_epoch": [],
               "val_time_per_epoch": [],
               "total_time": 0}

    total_start_time = time.time()

    best_val_f1 = 0
    epochs_no_improve = 0

    for epoch in tqdm(range(epochs)):
        train_start = time.time()
        train_loss, train_iou, train_f1 = train_step(model=model,
                                                     dataloader=train_dataloader,
                                                     loss_fn=loss_fn,
                                                     optimizer=optimizer,
                                                     device=device)
        train_end = time.time()

        val_start = time.time()
        val_loss, val_iou, val_f1 = val_step(model=model,
                                             dataloader=val_dataloader,
                                             loss_fn=loss_fn,
                                             device=device)
        val_end = time.time()

        epoch_train_time = train_end - train_start
        epoch_val_time = val_end - val_start

        print(
            f"Epoch {epoch + 1}/{epochs}\n"
            f"Train loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | Train F1: {train_f1:.4f} | Time: {epoch_train_time:.1f}s\n"
            f"Val loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val F1: {val_f1:.4f} | Time: {epoch_val_time:.1f}s\n")
        results["train_loss"].append(train_loss)
        results["train_iou"].append(train_iou)
        results["train_f1"].append(train_f1)
        results["val_loss"].append(val_loss)
        results["val_iou"].append(val_iou)
        results["val_f1"].append(val_f1)
        results["train_time_per_epoch"].append(epoch_train_time)
        results["val_time_per_epoch"].append(epoch_val_time)

        if writer:
            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={"train_loss": train_loss,
                                                "val_loss": val_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag="IoU",
                               tag_scalar_dict={"train_iou": train_iou,
                                                "val_iou": val_iou},
                               global_step=epoch)
            writer.add_scalars(main_tag="F1 Score",
                               tag_scalar_dict={"train_f1": train_f1,
                                                "val_f1": val_f1},
                               global_step=epoch)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0

            torch.save(obj=model.state_dict(), f=save_path)
            print(f"Val F1 improved: {best_val_f1:.4f}. Saved model to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")

        if epochs_no_improve >= patience:
            print(f"Early stopping at {epoch + 1}")
            break

    total_end_time = time.time()
    results["total_time"] = total_end_time - total_start_time

    if writer:
        writer.close()

    return results
