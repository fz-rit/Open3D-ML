import torch
checkpoint_path = "/home/fzhcis/data/semantic3d_full/selected/checkpoints/ckpt_00100.pth"
ckpt = torch.load(checkpoint_path, map_location="cpu")

# List the top-level keys
print(ckpt.keys())
# print(ckpt['config'])
print(ckpt['state_dict'].keys())
