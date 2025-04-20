import numpy as np
from torchvision import transforms
import torch


def preprocess(img):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  
            std=[0.229, 0.224, 0.225]    
        )
    ])
    img = trans(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img


def postprocess(prediction, shape):
    prediction_soft = torch.nn.functional.softmax(prediction, dim=1)
    prediction_max = torch.argmax(prediction_soft, dim=1)

    prediction_resized = transforms.functional.resize(
        prediction_max, size=shape, interpolation=transforms.InterpolationMode.NEAREST
    )

    prediction_numpy = prediction_resized.cpu().detach().numpy().squeeze()
    return prediction_numpy.astype(np.uint8)