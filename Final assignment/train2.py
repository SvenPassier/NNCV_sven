import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
)

from model import Model  # Import the modified Model class


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():
    parser = ArgumentParser("Training script for a PyTorch Model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="model-training", help="Experiment ID for Weights & Biases")

    return parser


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=args.experiment_id,
        config=vars(args),
    )

    # Create output directory
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set random seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms
    transform = Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ])

    # Load dataset
    train_dataset = Cityscapes(
        args.data_dir, split="train", mode="fine", target_type="semantic", transforms=transform
    )
    valid_dataset = Cityscapes(
        args.data_dir, split="val", mode="fine", target_type="semantic", transforms=transform
    )

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Define the model
    model = Model(in_channels=3, n_classes=19).to(device)  # Using the modified model with OOD classification

    # Define loss functions
    segmentation_loss_fn = nn.CrossEntropyLoss(ignore_index=255)  # Ignore void class for segmentation
    classification_loss_fn = nn.BCEWithLogitsLoss()  # Binary classification loss for OOD detection

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training phase
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()
            segmentation_output, classification_output = model(images)  # Get both outputs

            # Generate OOD labels (this part needs to be adapted based on actual OOD labeling)
            batch_size = classification_output.shape[0]
            ood_labels = torch.zeros(batch_size, 1, device=device)  # Assuming all are in-distribution for now

            # Compute losses
            segmentation_loss = segmentation_loss_fn(segmentation_output, labels)
            classification_loss = classification_loss_fn(classification_output, ood_labels)

            # Total loss (combine both)
            total_loss = segmentation_loss + classification_loss

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            wandb.log({
                "train_segmentation_loss": segmentation_loss.item(),
                "train_classification_loss": classification_loss.item(),
                "total_train_loss": total_loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            
        # Validation phase
        model.eval()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)

                segmentation_output, classification_output = model(images)

                # Generate dummy OOD labels for validation (to be replaced with real OOD labels)
                batch_size = classification_output.shape[0]
                ood_labels = torch.zeros(batch_size, 1, device=device)

                segmentation_loss = segmentation_loss_fn(segmentation_output, labels)
                classification_loss = classification_loss_fn(classification_output, ood_labels)

                total_loss = segmentation_loss + classification_loss
                losses.append(total_loss.item())
            
                if i == 0:
                    predictions = segmentation_output.softmax(1).argmax(1)
                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            wandb.log({"valid_loss": valid_loss}, step=(epoch + 1) * len(train_dataloader) - 1)

    print("Training complete!")
    wandb.finish()

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
