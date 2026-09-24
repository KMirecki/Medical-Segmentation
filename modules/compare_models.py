from pathlib import Path

import albumentations as A
import segmentation_models_pytorch as smp
import torch
from albumentations import ToTensorV2
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from modules.data_preparation import create_dataloaders
from modules.predict import compare_models_predictions


def load_model(
    model_cls, encoder_name: str, weights_path: Path, device: torch.device
) -> torch.nn.Module:
    model = model_cls(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path("models")
    dataset_path = Path("./datasets/HAM10000")

    preprocess_input = get_preprocessing_fn("resnet34", pretrained="imagenet")
    eval_transform = A.Compose(
        [
            A.Resize(256, 256),
            A.Lambda(image=preprocess_input),
            ToTensorV2(),
        ]
    )

    _, _, test_loader = create_dataloaders(
        dataset_path=dataset_path,
        train_transform=eval_transform,
        eval_transform=eval_transform,
        batch_size=16,
    )

    models_configs = [
        ("U-Net", smp.Unet, models_dir / "ham_Unet_efficientnet-b0_Adam.pth"),
        (
            "U-Net++",
            smp.UnetPlusPlus,
            models_dir / "ham_Unet++_efficientnet-b0_SGD.pth",
        ),
        (
            "DeepLab v3",
            smp.DeepLabV3,
            models_dir / "ham_Deeplabv3_efficientnet-b0_SGD.pth",
        ),
        (
            "DeepLab v3+",
            smp.DeepLabV3Plus,
            models_dir / "ham_Deeplabv3+_efficientnet-b0_SGD.pth",
        ),
        (
            "SegFormer",
            smp.Segformer,
            models_dir / "ham_Segformer_efficientnet-b0_SGD.pth",
        ),
    ]

    loaded_models = []
    for name, model_cls, path in models_configs:
        if not path.exists():
            print(f"No File: {path}. Skipping model {name}.")
            continue
        model = load_model(
            model_cls, encoder_name="efficientnet-b0", weights_path=path, device=device
        )
        loaded_models.append((name, model))

    if loaded_models:
        compare_models_predictions(
            models_with_names=loaded_models,
            dataloader=test_loader,
            device=device,
            seed=42,
        )
    else:
        print("No data found")


if __name__ == "__main__":
    main()
