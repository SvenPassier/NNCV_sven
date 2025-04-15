import torch

class MeanDice(torch.nn.Module):
    def __init__(self, num_classes=19, ignore_index=255, epsilon=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, pred, target):
        pred = torch.nn.functional.softmax(pred, dim=1)  # (batch, num_classes, H, W)
        valid = (target != self.ignore_index)

        # mask out the ignore_index
        target = target.masked_fill(~valid, 0)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()  # (batch, num_classes, H, W)
        target_one_hot = target_one_hot * valid.unsqueeze(1).float()

        # calculate mean dice score
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice_score = (2 * intersection + self.epsilon) / (union + self.epsilon)
        return 1 - dice_score.mean()