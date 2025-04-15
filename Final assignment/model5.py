import torch
from transformers import SegformerForSemanticSegmentation

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640",
            num_labels=19,
            ignore_mismatched_sizes=True
        )
        
    def forward(self, pixels, **kwargs):
        logits = self.segformer(pixels).logits
        return torch.nn.functional.interpolate(
            logits,
            size=pixels.shape[2:], 
            mode="bilinear",
            align_corners=False,
        )