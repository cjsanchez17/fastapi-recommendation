import torch
import numpy as np

# Load the PyTorch tensor
tensor_data = torch.load("tag_vector.pt", weights_only=True)

# Convert to numpy and save as .npy
np.save("tag_vector.npy", tensor_data.numpy())
