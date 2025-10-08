import torch
checkpoint_path = "/home/fzhcis/data/semantic3d_full/selected/checkpoints/ckpt_00100.pth"
ckpt = torch.load(checkpoint_path, map_location="cpu")

# List the top-level keys
state_dict = ckpt["model_state_dict"]

print(list(state_dict.keys())[:20])  # show first 20 keys

# Find the first and last weight tensors
first_key = list(state_dict.keys())[0]
last_key = list(state_dict.keys())[-1]

print(first_key, state_dict[first_key].shape)
print(last_key, state_dict[last_key].shape)
