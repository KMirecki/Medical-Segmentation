import torch
from albumentations import ToTensorV2
from pathlib import Path
from modules.helper_functions import walk_through_dir, display_random_images, plot_loss_curves, dataloaders_info, \
    create_writer
from modules.data_preparation import create_dataloaders
from modules.model_training import train
from modules.predict import predict, visualize_predictions, visualize_single_prediction, compare_models_predictions
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import albumentations as A
from torchinfo import summary
import pandas as pd
import os


def main():
    print(torch.__version__)
    # Device agnostic code
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Setting paths to datasets
    data_path = Path("./datasets")
    brisc_dataset_path = data_path / "brisc2025"
    bus_uc_dataset_path = data_path / "BUS_UC"
    ham_dataset_path = data_path / "HAM10000"

    # Print datasets info
    walk_through_dir(brisc_dataset_path)
    walk_through_dir(bus_uc_dataset_path)
    walk_through_dir(ham_dataset_path)

    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH = Path("results")
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    global_test_csv_path = RESULTS_PATH / "tests.csv"

    IMG_SIZE = 256
    preprocess_input = get_preprocessing_fn(encoder_name='resnet34', pretrained='imagenet')

    train_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Lambda(image=preprocess_input),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Lambda(image=preprocess_input),
        ToTensorV2()
    ])

    BATCH_SIZE = 16

    ham_train_loader, ham_val_loader, ham_test_loader = create_dataloaders(dataset_path=ham_dataset_path,
                                                                           train_transform=train_transform,
                                                                           test_transform=test_transform,
                                                                           batch_size=BATCH_SIZE)

    # dataloaders_info(ham_train_loader, ham_val_loader, ham_test_loader, BATCH_SIZE)
    # display_random_images(ham_train_loader)
    #
    # models = ["Segformer"]
    # encoders = ["mobilenet_v2"]
    # optimizers = ["AdamW"]
    #
    # EPOCHS = 50
    # for model_name in models:
    #     for encoder_name in encoders:
    #         for optimizer_name in optimizers:
    #             print("-" * 50 + "\n")
    #
    #             match (model_name):
    #                 case "Unet":
    #                     model = smp.Unet(encoder_name=encoder_name,
    #                                      encoder_weights="imagenet",
    #                                      in_channels=3,
    #                                      classes=1).to(DEVICE)
    #                 case "Unet++":
    #                     model = smp.UnetPlusPlus(encoder_name=encoder_name,
    #                                              encoder_weights="imagenet",
    #                                              in_channels=3,
    #                                              classes=1).to(DEVICE)
    #                 case "Deeplabv3":
    #                     model = smp.DeepLabV3(encoder_name=encoder_name,
    #                                           encoder_weights="imagenet",
    #                                           in_channels=3,
    #                                           classes=1).to(DEVICE)
    #                 case "Deeplabv3+":
    #                     model = smp.DeepLabV3Plus(encoder_name=encoder_name,
    #                                               encoder_weights="imagenet",
    #                                               in_channels=3,
    #                                               classes=1).to(DEVICE)
    #                 case "Segformer":
    #                     model = smp.Segformer(encoder_name=encoder_name,
    #                                           encoder_weights="imagenet",
    #                                           in_channels=3,
    #                                           classes=1).to(DEVICE)
    #
    #             loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)
    #
    #             if optimizer_name == "SGD":
    #                 optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, momentum=0.9)
    #             elif optimizer_name == "Adam":
    #                 optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    #             elif optimizer_name == "AdamW":
    #                 optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001)
    #
    #             train_results = train(model=model,
    #                                   train_dataloader=ham_train_loader,
    #                                   val_dataloader=ham_val_loader,
    #                                   loss_fn=loss_fn,
    #                                   optimizer=optimizer,
    #                                   epochs=EPOCHS,
    #                                   device=DEVICE,
    #                                   writer=create_writer(dataset_name="ham",
    #                                                        model_name=model_name,
    #                                                        extra=f"{encoder_name}_{optimizer_name}"),
    #                                   save_path=f"{MODEL_PATH}/ham_{model_name}_{encoder_name}_{optimizer_name}.pth")
    #
    #             train_csv_filename = f"train_ham_{model_name}_{encoder_name}_{optimizer_name}.csv"
    #             train_csv_path = RESULTS_PATH / train_csv_filename
    #
    #             total_time = train_results.pop("total_time", 0)
    #
    #             train_df = pd.DataFrame(train_results)
    #             train_df.to_csv(train_csv_path, index=False)
    #
    #             plot_loss_curves(results=train_results,
    #                              title=f"ham_{model_name}_{encoder_name}_{optimizer_name}")
    #
    #             test_results = predict(model=model,
    #                                    dataloader=ham_test_loader,
    #                                    device=DEVICE)
    #             test_results["model_name"] = model_name
    #             test_results["encoder_name"] = encoder_name
    #             test_results["optimizer"] = optimizer_name
    #             test_results["dataset_name"] = "ham"
    #
    #             test_df = pd.DataFrame([test_results])
    #
    #             if not os.path.isfile(global_test_csv_path):
    #                 test_df.to_csv(global_test_csv_path, index=False)
    #             else:
    #                 test_df.to_csv(global_test_csv_path, mode='a', header=False, index=False)
    #
    #             visualize_predictions(model=model,
    #                                   dataloader=ham_test_loader,
    #                                   device=DEVICE)
    model_unet = smp.Unet(encoder_name="efficientnet-b0",
                          encoder_weights="imagenet",
                          in_channels=3,
                          classes=1)

    model_unet.load_state_dict(torch.load(MODEL_PATH / "ham_Unet_efficientnet-b0_Adam.pth"))
    model_unet.to(DEVICE)

    model_unetplusplus = smp.UnetPlusPlus(encoder_name="efficientnet-b0",
                                          encoder_weights="imagenet",
                                          in_channels=3,
                                          classes=1)

    model_unetplusplus.load_state_dict(torch.load(MODEL_PATH / "ham_Unet++_efficientnet-b0_SGD.pth"))
    model_unetplusplus.to(DEVICE)

    model_deeplab = smp.DeepLabV3(encoder_name="efficientnet-b0",
                                  encoder_weights="imagenet",
                                  in_channels=3,
                                  classes=1)

    model_deeplab.load_state_dict(torch.load(MODEL_PATH / "ham_Deeplabv3_efficientnet-b0_SGD.pth"))
    model_deeplab.to(DEVICE)

    model_deeplabplus = smp.DeepLabV3Plus(encoder_name="efficientnet-b0",
                                          encoder_weights="imagenet",
                                          in_channels=3,
                                          classes=1)

    model_deeplabplus.load_state_dict(torch.load(MODEL_PATH / "ham_Deeplabv3+_efficientnet-b0_SGD.pth"))
    model_deeplabplus.to(DEVICE)

    model_segformer = smp.Segformer(encoder_name="efficientnet-b0",
                                    encoder_weights="imagenet",
                                    in_channels=3,
                                    classes=1)

    model_segformer.load_state_dict(torch.load(MODEL_PATH / "ham_Segformer_efficientnet-b0_SGD.pth"))
    model_segformer.to(DEVICE)

    models = [
        ("U-Net", model_unet),
        ("U-Net++", model_unetplusplus),
        ("DeepLab v3", model_deeplab),
        ("DeepLab v3+", model_deeplabplus),
        ("SegFormer", model_segformer)
    ]

    compare_models_predictions(
        models_with_names=models,
        dataloader=ham_test_loader,
        device=DEVICE,
        seed=51
    )


if __name__ == '__main__':
    main()
