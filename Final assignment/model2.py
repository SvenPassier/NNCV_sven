from transformers import SegformerForSemanticSegmentation
import torch

"""
load the NVIDIA SegFormer B5 model from huggingface: 
https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640
"""

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-ade-640-640",
            num_labels=19,
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        logits = self.segformer(x).logits
        return torch.nn.functional.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)