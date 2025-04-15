import os
from argparse import ArgumentParser
import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (Compose, Normalize, ToImage,ToDtype, RandomHorizontalFlip, RandomCrop, RandomAffine, ColorJitter, GaussianBlur)
from torch.cuda.amp import GradScaler, autocast
from torchvision.tv_tensors import Image, Mask

from mappings import convert_to_train_id, convert_train_id_to_color, visualize_result
from model import Model
from losses import MeanDice

def get_args_parser():
    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Path to checkpoint for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Number of mini-batches to accumulate gradients over")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="Weight decay for the optimizer")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser

def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability, if you add other sources of randomness (NumPy, Random), make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transforms to apply to the data
    train_transform_robustness = Compose([
        ToImage(),
        RandomAffine(degrees=0, scale=(0.5, 2.0)),
        RandomCrop((1024, 1024), pad_if_needed=True, fill={Image: 0, Mask: 255}),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)), # imagenet values (used in ade20k training)
    ])

     # Define the transforms to apply to the data
    train_transform = Compose([
        ToImage(),
        RandomCrop((1024, 1024)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)), # imagenet values (used in ade20k training)
    ])

    val_transform = Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)), # imagenet values (used in ade20k training)
    ])

    # Load the dataset and make a split for training and validation
    train_dataset = Cityscapes(
        args.data_dir, 
        split="train", 
        mode="fine", 
        target_type="semantic", 
        transforms=train_transform
    )

    valid_dataset = Cityscapes(
        args.data_dir, 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transforms=val_transform
    )

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Define the model
    model = Model().to(device)

    # load pretrained weigths checkpoint
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint)

    # Define the loss function and optimizer
    criterion = MeanDice()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # poly decay
    lr_lambda = lambda epoch: (1 - epoch / args.epochs) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # initialize AMP GradScaler
    scaler = torch.cuda.amp.GradScaler()

    # number of gradient accumulation steps
    accumulation_steps = args.accumulation_steps

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()

        # zero gradient and accumulated loss at start of epoch
        optimizer.zero_grad()

        losses = []

        for i, (images, labels) in enumerate(train_dataloader):
            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)  # Remove channel dimension
            
            # Forward pass with autocast
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            losses.append(loss)
            loss = loss / accumulation_steps

            # Backward pass with scaling
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                loss = sum(losses)/len(losses)

                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch + 1,
                })
                losses = []

        scheduler.step()
            
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                # Use autocast during inference
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                losses.append(loss.item())

                # visualize result in wandb
                if i == 0:
                    predictions_img, labels_img = visualize_result(outputs, labels)
                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, commit=False)
            
            valid_loss = sum(losses) / len(losses)

            wandb.log({"valid_loss": valid_loss}, commit=False)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"epoch={epoch:05}-val_loss={valid_loss:05}.pth"
                )
                torch.save(model.state_dict(), current_best_model_path)

    # Save the model
    torch.save(model.state_dict(), os.path.join(output_dir, f"epoch={epoch:05}-val_loss={valid_loss:05}.pth"))
    wandb.finish()

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)