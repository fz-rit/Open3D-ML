#!/usr/bin/env python3
import sys
from pathlib import Path

# Add repo root
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

print(f"Python path: {sys.path[0]}")

try:
    print("Importing ml3d.torch.models.randlanet_contrast...")
    import ml3d.torch.models.randlanet_contrast as m
    print(f"Module loaded: {m}")
    print(f"Module file: {m.__file__}")
    print(f"\nModule attributes: {[x for x in dir(m) if not x.startswith('_')]}")
    print(f"\nRandLANetContrast: {m.RandLANetContrast}")
    print(f"Type: {type(m.RandLANetContrast)}")
    
    if m.RandLANetContrast is None:
        print("\n ERROR: RandLANetContrast is None!")
        print("This means the class definition failed silently.")
        
except Exception as e:
    print(f"Error importing: {e}")
    import traceback
    traceback.print_exc()
