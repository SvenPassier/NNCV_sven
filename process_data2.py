import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

from torchvision.datasets import Cityscapes


id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}

def convert_to_train_id(prediction):
    return np.vectorize(lambda x: id_to_trainid.get(x, 255))(prediction)  # Default to 255 for ignored labels


def preprocess(img):

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to match model input
        transforms.ToTensor(),          # Convert PIL image to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  
    ])

    img = transform(img)  
    img = img.unsqueeze(0)  
    return img


def postprocess(prediction, shape):


    if not isinstance(prediction, torch.Tensor):
        prediction = torch.tensor(prediction)

    prediction = prediction.softmax(dim=1).argmax(dim=1)  # [batch, classes, h, w] → [batch, h, w]
    prediction = F.interpolate(prediction.unsqueeze(1).float(), size=shape, mode="nearest").squeeze(1)
    prediction_np = prediction.cpu().numpy().astype(np.uint8)  # Convert to uint8 format
    prediction_np = convert_to_train_id(prediction_np)

    return prediction_np  
