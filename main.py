from pathlib import Path

import albumentations as A
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from albumentations import ToTensorV2
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from modules.data_preparation import create_dataloaders
from modules.helper_functions import create_writer, plot_training_curves
from modules.model_training import train
from modules.predict import predict


def get_model(
    model_name: str, encoder_name: str, in_channels: int = 3, classes: int = 1
) -> torch.nn.Module:
    model_cls = {
        "Unet": smp.Unet,
        "Unet++": smp.UnetPlusPlus,
        "Deeplabv3": smp.DeepLabV3,
        "Deeplabv3+": smp.DeepLabV3Plus,
        "Segformer": smp.Segformer,
    }.get(model_name)

    if model_cls is None:
        raise ValueError(f"Unknown architecture: {model_name}")

    return model_cls(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes,
    )


def get_optimizer(
    optimizer_name: str, model_params, lr: float = 1e-3
) -> torch.optim.Optimizer:
    if optimizer_name == "SGD":
        return torch.optim.SGD(params=model_params, lr=0.1, momentum=0.9)
    if optimizer_name == "Adam":
        return torch.optim.Adam(params=model_params, lr=lr)
    if optimizer_name == "AdamW":
        return torch.optim.AdamW(params=model_params, lr=lr)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch: {torch.__version__} | Device: {device}")

    dataset_path = Path("./datasets/HAM10000")
    dataset_name = "ham"

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    global_test_csv = results_dir / "tests.csv"

    img_size = 256
    batch_size = 16
    epochs = 50
    patience = 5

    preprocess_input = get_preprocessing_fn("resnet34", pretrained="imagenet")

    train_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Lambda(image=preprocess_input),
            ToTensorV2(),
        ]
    )

    eval_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Lambda(image=preprocess_input),
            ToTensorV2(),
        ]
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_path=dataset_path,
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=batch_size,
    )

    models = ["Unet", "Unet++", "Deeplabv3", "Deeplabv3+", "Segformer"]
    encoders = ["resnet34", "efficientnet-b0", "mobilenet_v2"]
    optimizers = ["SGD", "Adam", "AdamW"]

    loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)

    for model_name in models:
        for encoder_name in encoders:
            for opt_name in optimizers:
                run_tag = f"{dataset_name}_{model_name}_{encoder_name}_{opt_name}"
                model_save_path = str(models_dir / f"{run_tag}.pth")

                print("\n" + "=" * 60)
                print(f"Start: {run_tag}")
                print("=" * 60)

                model = get_model(model_name, encoder_name).to(device)
                optimizer = get_optimizer(opt_name, model.parameters())
                writer = create_writer(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    extra=f"{encoder_name}_{opt_name}",
                )

                train_results = train(
                    model=model,
                    train_dataloader=train_loader,
                    val_dataloader=val_loader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    epochs=epochs,
                    device=device,
                    writer=writer,
                    save_path=model_save_path,
                    patience=patience,
                )

                total_time = train_results.pop("total_time", 0)
                train_df = pd.DataFrame(train_results)
                train_df.to_csv(results_dir / f"train_{run_tag}.csv", index=False)

                plot_training_curves(results=train_results, title=run_tag)

                model.load_state_dict(
                    torch.load(model_save_path, map_location=device, weights_only=True)
                )

                test_results = predict(
                    model=model, dataloader=test_loader, device=device
                )
                test_results.update(
                    {
                        "model_name": model_name,
                        "encoder_name": encoder_name,
                        "optimizer": opt_name,
                        "dataset_name": dataset_name,
                        "total_time": total_time,
                    }
                )

                test_df = pd.DataFrame([test_results])
                write_header = not global_test_csv.exists()
                test_df.to_csv(
                    global_test_csv,
                    mode="a",
                    header=write_header,
                    index=False,
                )

                print(
                    f"Test F1: {test_results['test_f1']:.4f} | Test IoU: {test_results['test_iou']:.4f}"
                )


if __name__ == "__main__":
    main()
