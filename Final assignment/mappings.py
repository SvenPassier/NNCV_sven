import torch
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid

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

def visualize_result(outputs, labels):
    predictions = outputs.softmax(1).argmax(1)
    predictions = predictions.unsqueeze(1)
    labels = labels.unsqueeze(1)
    predictions = convert_train_id_to_color(predictions)
    labels = convert_train_id_to_color(labels)
    predictions_img = make_grid(predictions.cpu(), nrow=8)
    labels_img = make_grid(labels.cpu(), nrow=8)
    predictions_img = predictions_img.permute(1, 2, 0).numpy()
    labels_img = labels_img.permute(1, 2, 0).numpy()
    return predictions_img, labels_img