import os
from argparse import ArgumentParser
from typing import Dict
import numpy as np

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
    Pad,
    RandomHorizontalFlip,
    RandomRotation,
    RandomResizedCrop,
    GaussianBlur,
    InterpolationMode,
    ColorJitter
)

from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, MultiStepLR
from model import Model
from dino_model import DINOv2Segmentation
# from torchinfo import summary

# Mean and standard deviation of the Cityscapes dataset
MEAN = [0.28689554, 0.32513303, 0.28389177]
STD = [0.18696375, 0.19017339, 0.18720214]

# Cityscapes classes
CITYSCAPES_CLASSES = [cls.name for cls in Cityscapes.classes if cls.train_id != 255]
print("Cityscapes classes:", CITYSCAPES_CLASSES)

# Pre-calculated class weights for Cityscapes
CITYSCAPES_CLASS_WEIGHTS = torch.tensor([
    1.000,  # road              (41.70%)
    2.882,  # sidewalk          (4.04%)
    1.438,  # building          (20.51%)
    4.598,  # wall              (0.44%)
    4.421,  # fence             (0.64%)
    3.979,  # pole              (1.24%)
    4.841,  # traffic light     (0.21%)
    4.483,  # traffic sign      (0.57%)
    1.636,  # vegetation        (15.70%)
    4.262,  # terrain           (0.83%)
    2.633,  # sky               (5.19%)
    3.891,  # person            (1.38%)
    4.904,  # rider             (0.15%)
    2.461,  # car               (6.17%)
    4.767,  # truck             (0.28%)
    4.773,  # bus               (0.27%)
    4.828,  # train             (0.22%)
    5.000,  # motorcycle        (0.07%)
    4.639   # bicycle           (0.40%)
])

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

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="Weight decay for AdamW optimizer")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    parser.add_argument("--model", type=str, default="unet", help="Model architecture to use")
    parser.add_argument("--weighted", action="store_true", help="Use class weights in loss function")
    parser.add_argument("--multistep", action="store_true", help="Use MultiStepLR scheduler")
    parser.add_argument("--onecycle", action="store_true", help="Use OneCycleLR scheduler")
    parser.add_argument("--scheduler-epochs", type=int, nargs="+", default=[30], help="Epochs to adjust learning rate")
    parser.add_argument("--fine-tune", action="store_true", help="Fine-tune the model")
    
    return parser
    

def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 19, smooth: float = 1e-6) -> Dict[int, float]:
    """
    Calculate Dice score for each class
    Args:
        pred: Predictions tensor after softmax (B, C, H, W)
        target: Ground truth tensor (B, H, W)
        num_classes: Number of classes
        smooth: Smoothing factor to avoid division by zero
    Returns:
        Dictionary of class-wise Dice scores
    """
    dice_scores = {}
    
    for cls in range(num_classes):
        pred_cls = pred[:, cls]  # (B, H, W)
        target_cls = (target == cls).float()  # (B, H, W)
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores[cls] = dice.item()
    
    return dice_scores


def calculate_class_weights(dataset):
    """Calculate inverse frequency class weights with moderate emphasis on rare classes"""
    class_counts = torch.zeros(19)
    total_pixels = 0
    print("Calculating class weights...")
    
    # Count occurrences
    for _, label in dataset:
        label = convert_to_train_id(label).cpu()
        valid_mask = (label < 19)
        label_valid = label[valid_mask]
        classes, counts = torch.unique(label_valid, return_counts=True)
        for cls, cnt in zip(classes, counts):
            if cls < 19:
                class_counts[cls.long()] += cnt
                total_pixels += cnt
    
    # Calculate frequencies
    frequencies = class_counts / total_pixels
    
    # Calculate weights with sqrt to moderate the inverse relationship
    weights = 1.0 / torch.sqrt(frequencies + 0.02)
    
    # Normalize weights to range from 1 to ~5
    weights = ((weights - weights.min()) / (weights.max() - weights.min()) * 4) + 1.0
    
    # Print statistics
    print("\nClass Statistics:")
    print(f"{'Class':<20} {'Count':<12} {'Freq %':<10} {'Weight':<10}")
    print("-" * 55)
    
    for i in range(19):
        count = class_counts[i].item()
        freq = frequencies[i].item() * 100
        weight = weights[i].item()
        print(f"{CITYSCAPES_CLASSES[i]:<20} {count:<12.0f} {freq:<10.2f} {weight:<10.3f}")
    
    print(f"\nTotal pixels: {total_pixels:,}")
    print(f"Weight range: {weights.min().item():.3f} to {weights.max().item():.3f}")
    
    return weights


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

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Define the transforms to apply to the train images
    train_transform = Compose([
        ToImage(),
        RandomResizedCrop(
            size=(640, 640), 
            scale=(0.5, 2.0),
            ratio=(0.75, 1.333)),
        Pad(padding=[2, 2, 2, 2], padding_mode='constant', fill=0),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=(-10, 10),
                       interpolation=InterpolationMode.BILINEAR,
                       fill=0),
        ColorJitter(
            brightness=0.3, 
            contrast=0.3, 
            saturation=0.3, 
            hue=0.1),
        GaussianBlur(kernel_size=(3, 3), 
                     sigma=(0.1, 1.0)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=MEAN, std=STD),
    ])

    # Validation transform without augmentations
    valid_transform = Compose([
        ToImage(),
        Resize(size=(640, 640)),
        Pad(padding=[2, 2, 2, 2], padding_mode='constant', fill=0),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=MEAN, std=STD),
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
        transforms=valid_transform
    )

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    # Print dataset sizes
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")

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
    
    ################# DEFINE THE MODEL ARCHITECTURE ##################
    if args.model == "unet":
        print("Initializing U-Net model")
        model = Model(
            in_channels=3,  # RGB images
            n_classes=19,  # 19 classes in the Cityscapes dataset
        ).to(device)

        # Define the optimizer
        optimizer = AdamW(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

    elif args.model == "dinov2":
        print("Initializing DINOv2 backbone")
        model = DINOv2Segmentation(num_classes=19, fine_tune=args.fine_tune)
        model.to(device)
        
        # Define the optimizer
        optimizer = AdamW(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

        if args.multistep:
            # MultiStepLR scheduler
            scheduler = MultiStepLR(
                optimizer,
                milestones=args.scheduler_epochs,
                gamma=0.1
            )
        elif args.onecycle:
            # OneCycleLR scheduler
            scheduler = OneCycleLR(
                optimizer,
                max_lr=args.lr,  # Peak learning rate
                epochs=args.epochs,
                steps_per_epoch=len(train_dataloader),
                pct_start=0.1,  # Spend 10% of time warming up
                div_factor=10,
                final_div_factor=100,  # Final LR = max_lr/50
                anneal_strategy='cos' # Cosine annealing
            )
    else:
        raise ValueError(f"Invalid model type: {args.model}")
    
    # Define the loss function
    if args.weighted:
        print("Using pre-calculated class weights")
        class_weights = CITYSCAPES_CLASS_WEIGHTS.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class

    ##################################################################

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()
            outputs = model(images)

            ##### BATCH-WISE LOSS #####
            loss = criterion(outputs, labels)
            #############################

            ##### BACKPROPAGATION AND PARAMETER UPDATION #####
            loss.backward()
            optimizer.step()
            ##################################################

            # OneCycleLR scheduler step
            if args.onecycle:
                scheduler.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            epoch_dice_scores = []

            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                # Convert to long and remove channel dimension
                labels = labels.long().squeeze(1)  # [B, 640, 640] 

                outputs = model(images) # [B, 46, 46]

                # Calculate loss
                loss = criterion(outputs, labels)
                losses.append(loss.item())

                # Calculate Dice Score
                dice_preds = outputs.softmax(1) 
                batch_dice_scores = calculate_dice_score(dice_preds, labels)
                epoch_dice_scores.append(batch_dice_scores)

                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=16)
                    labels_img = make_grid(labels.cpu(), nrow=16)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)

            # Calculate mean Dice score for the epoch
            class_dice_scores = {}
            for cls in range(19):
                class_dice_scores[cls] = np.mean([dice[cls] for dice in epoch_dice_scores])
            
            mean_dice = np.mean(list(class_dice_scores.values()))

            # If MultiStepLR scheduler is used, step at the end of each epoch
            if args.multistep:
                scheduler.step()

            wandb.log({
                "valid_loss": valid_loss,
                "mean_dice_score": mean_dice,
                **{f"dice_{CITYSCAPES_CLASSES[cls]}": score for cls, score in class_dice_scores.items()},
            }, step=(epoch + 1) * len(train_dataloader) - 1) # Log at the end of the epoch

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
                )
                torch.save(model.state_dict(), current_best_model_path)

    print("TRAINING COMPLETE!")

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