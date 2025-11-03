#!/usr/bin/env python
"""Quick script to inspect checkpoint structure"""
import torch
import sys

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'logs/checkpoint/ckpt_best.pth'

print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu')

print(f"\nCheckpoint keys: {list(ckpt.keys())}")

if 'model_state_dict' in ckpt:
    sd = ckpt['model_state_dict']
    print(f"\nTotal parameters: {len(sd)}")
    
    # Group by prefix
    prefixes = {}
    for key in sd.keys():
        prefix = key.split('.')[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    
    print("\nParameter groups:")
    for prefix, count in sorted(prefixes.items()):
        print(f"  {prefix}: {count} params")
    
    # Check projection head shape
    proj_keys = [k for k in sd.keys() if 'projection' in k]
    if proj_keys:
        print(f"\nProjection head keys ({len(proj_keys)} total):")
        for k in proj_keys[:3]:
            print(f"  {k}: {sd[k].shape}")
