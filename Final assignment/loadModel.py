import torch
from model import Model          

"""
Load the model from the wrapped model for CodaLab submission
"""

full_path   = ""
state_path  = "model.pth"        

model = torch.load(full_path, map_location="cpu")

if isinstance(model, torch.nn.DataParallel):
    model = model.module

torch.save(model.state_dict(), state_path)
print("state‑dict written to", state_path)

