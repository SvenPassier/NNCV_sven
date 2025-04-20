import os
from argparse import ArgumentParser
import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torch.cuda.amp import GradScaler, autocast
from torchvision.tv_tensors import Image, Mask
from torchvision.transforms.v2 import (Compose, Normalize, ToImage,ToDtype, RandomHorizontalFlip,
                                       RandomCrop, RandomAffine, ColorJitter, RandomRotation,
                                       RandomGrayscale, RandomSolarize, GaussianBlur)


from labelProcessing import convert_to_train_id, convert_train_id_to_color
from lossDice import DiceLoss
from model2 import Model


def get_args_parser():
    parser = ArgumentParser("Training script parser")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    parser.add_argument("--accumulation", type=int, default=2, help="Accumulate gradients for steps")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="Reduce large weights")
    return parser


def render_img(pred_logits, gt_labels):
    
    predicted = torch.argmax(torch.softmax(pred_logits, dim=1), dim=1).unsqueeze(1)
    ground_truth = gt_labels.unsqueeze(1)

    pred_color = convert_train_id_to_color(predicted)
    gt_color = convert_train_id_to_color(ground_truth)

    pred_grid = make_grid(pred_color.cpu(), nrow=4)
    gt_grid = make_grid(gt_color.cpu(), nrow=4)

    pred_vis = pred_grid.permute(1, 2, 0).numpy()
    gt_vis = gt_grid.permute(1, 2, 0).numpy()
    return pred_vis, gt_vis


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",   
        name=args.experiment_id,                   
        config=vars(args),                          
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability, if you add other sources of randomness (NumPy, Random), make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transforms to apply to the training data
    transform1 = Compose([
        ToImage(),
        RandomAffine(degrees=0, scale=(0.8, 1.2)),
        RandomRotation(degrees=5),
        RandomHorizontalFlip(p=0.5),
        RandomCrop((1024, 1024), pad_if_needed=True, fill={Image: 0, Mask: 255}),

        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        RandomGrayscale(p=0.1),
        # RandomSolarize(threshold=0.5, p=0.1)

        ToDtype(torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)),
    ])

    # Define transformation to apply to the validation data
    transform2 = Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)), 
    ])

    # Load the dataset and make a split for training and validation
    train_dataset = Cityscapes(
        args.data_dir, 
        split="train", 
        mode="fine", 
        target_type="semantic", 
        transforms=transform1
    )

    valid_dataset = Cityscapes(
        args.data_dir, 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transforms=transform2
    )

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
    
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers)

    # Define the model
    model = Model().to(device)
    if torch.cuda.device_count() > 1:
      print(f"Using {torch.cuda.device_count()} GPUs")
      model = torch.nn.DataParallel(model)

    # Define the loss function
    criterion = DiceLoss(num_classes=19, ignore_index=255)

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Define the scaler
    accumulation = args.accumulation
    grad_scaler = torch.cuda.amp.GradScaler()

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        losses = []
        model.train()
        optimizer.zero_grad()

        for i, (images, labels) in enumerate(train_dataloader):
            labels = convert_to_train_id(labels)           
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)  
            
            with torch.cuda.amp.autocast():
                # print(outputs.shape)
                outputs = model(images)
                loss = criterion(outputs, labels)

            losses.append(loss)
            loss = loss / accumulation
            grad_scaler.scale(loss).backward()

            # perform the optimizer step at every N batches or on the final batch
            if (i + 1) == len(train_dataloader) or (i + 1) % accumulation == 0:
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad()
                loss = sum(losses)/len(losses)

                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch + 1,
                })
                losses = []

            
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)  

                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                losses.append(loss.item())

                if i == 0:
                    predictions_img, labels_img = render_img(outputs, labels)
                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, commit=False)
            
            valid_loss = sum(losses) / len(losses)
            wandb.log({"valid_loss": valid_loss})

            # Save the model          
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, f"wrapped_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth")
                torch.save(model, current_best_model_path)

    torch.save(model.state_dict(), os.path.join(output_dir, f"epoch={epoch:04}-val_loss={valid_loss:04}.pth"))
    wandb.finish()



if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)