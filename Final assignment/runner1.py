from PIL import Image
from process_data import preprocess, postprocess
import torch

# Load a sample image
img = Image.open("path_to_sample_cityscapes_image.png")

# Preprocess the image
input_tensor = preprocess(img)
print("Preprocessed image shape:", input_tensor.shape)  # Expected: [1, 3, 256, 256]

# Simulate a model output
dummy_output = torch.randn(1, 19, 256, 256)  # Simulate model output logits

# Postprocess the output
original_shape = img.size[::-1]  # (height, width)
processed_output = postprocess(dummy_output, original_shape)
print("Postprocessed output shape:", processed_output.shape)