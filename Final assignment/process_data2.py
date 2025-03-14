import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

# Cityscapes mapping (from train.py)
from torchvision.datasets import Cityscapes

id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}

def convert_to_train_id(prediction):
    """ Converts raw model predictions to Cityscapes training IDs. """
    return np.vectorize(lambda x: id_to_trainid.get(x, 255))(prediction)  # Default to 255 for ignored labels


def preprocess(img):
    """
    Preprocess function to convert a PIL image to a model-compatible tensor.

    Args:
        img (PIL.Image): Input image in original size.

    Returns:
        torch.Tensor: Preprocessed image tensor (normalized, resized).
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to match model input
        transforms.ToTensor(),          # Convert PIL image to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize (same as train.py)
    ])

    img = transform(img)  # Apply transformations
    img = img.unsqueeze(0)  # Add batch dimension
    return img


def postprocess(prediction, shape):
    """
    Postprocess function to convert model output into a NumPy array with the original image shape.

    Args:
        prediction (torch.Tensor): Raw model output (logits or probabilities).
        shape (tuple): Original image shape (height, width).

    Returns:
        np.ndarray: Postprocessed array of shape (X, Y, n), where `n` is the number of class labels.
    """
    # Ensure prediction is a tensor
    if not isinstance(prediction, torch.Tensor):
        prediction = torch.tensor(prediction)

    # Convert logits to class labels
    prediction = prediction.softmax(dim=1).argmax(dim=1)  # [batch, classes, h, w] → [batch, h, w]

    # Resize back to original image shape
    prediction = F.interpolate(prediction.unsqueeze(1).float(), size=shape, mode="nearest").squeeze(1)

    # Convert to NumPy array
    prediction_np = prediction.cpu().numpy().astype(np.uint8)  # Convert to uint8 format

    # Convert to Cityscapes training IDs
    prediction_np = convert_to_train_id(prediction_np)

    return prediction_np  # Shape: (X, Y, n)
