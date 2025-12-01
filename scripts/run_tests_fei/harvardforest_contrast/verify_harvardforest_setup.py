#!/usr/bin/env python
"""
Verification script for HarvardForest3D SSL setup.

Checks:
1. Required Python packages installed
2. Dataset path exists and contains LAS files
3. Pretrained checkpoint exists (optional)
4. Config file is valid
5. Module imports work

Usage:
    python scripts/verify_harvardforest_setup.py --cfg ml3d/configs/harvardforest3d_ssl.yml
"""

import argparse
import sys
from pathlib import Path

# Add Open3D-ML to path (use local repo, not installed package)
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import ml3d.utils as utils
import ml3d.datasets as datasets
import ml3d.torch.models as models
import ml3d.torch.pipelines as pipelines

def check_packages():
    """Check if required packages are installed."""
    print("\n" + "="*60)
    print("Checking Python Packages")
    print("="*60)
    
    required = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'laspy': 'laspy (for LAS files)',
        'tqdm': 'tqdm',
        'sklearn': 'scikit-learn',
    }
    
    missing = []
    for package, name in required.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT FOUND")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    else:
        print("\n✓ All required packages installed")
        return True


def check_modules():
    """Check if Open3D-ML modules can be imported."""
    print("\n" + "="*60)
    print("Checking Open3D-ML Modules")
    print("="*60)
    
    try:
        from ml3d.datasets import HarvardForest3D
        print("✓ HarvardForest3D dataset")
    except ImportError as e:
        print(f"✗ HarvardForest3D dataset - {e}")
        return False
    
    try:
        from ml3d.torch.models import RandLANetSSL
        print("✓ RandLANetSSL model")
    except ImportError as e:
        print(f"✗ RandLANetSSL model - {e}")
        return False
    
    try:
        from ml3d.torch.pipelines import SSLRotation
        print("✓ SSLRotation pipeline")
    except ImportError as e:
        print(f"✗ SSLRotation pipeline - {e}")
        return False
    
    print("\n✓ All Open3D-ML modules loaded successfully")
    return True


def check_config(cfg_path):
    """Check if config file exists and is valid."""
    print("\n" + "="*60)
    print("Checking Configuration File")
    print("="*60)
    
    cfg_path = Path(cfg_path)
    
    if not cfg_path.exists():
        print(f"✗ Config file not found: {cfg_path}")
        return False, None
    
    print(f"✓ Config file exists: {cfg_path}")
    
    try:
        cfg = utils.Config.load_from_file(str(cfg_path))
        print("✓ Config file is valid YAML")
        return True, cfg
    except Exception as e:
        print(f"✗ Config file error: {e}")
        return False, None


def check_dataset(cfg):
    """Check if dataset path exists and contains LAS files."""
    print("\n" + "="*60)
    print("Checking Dataset")
    print("="*60)
    
    if cfg is None:
        print("✗ Cannot check dataset (config not loaded)")
        return False
    
    dataset_path = Path(cfg.dataset.dataset_path)
    
    if not dataset_path.exists():
        print(f"✗ Dataset path does not exist: {dataset_path}")
        print("  → Update 'dataset.dataset_path' in config file")
        return False
    
    print(f"✓ Dataset path exists: {dataset_path}")
    
    las_files = list(dataset_path.glob('*.las'))
    
    if len(las_files) == 0:
        print(f"✗ No .las files found in {dataset_path}")
        print("  → Check dataset path or file extensions")
        return False
    
    print(f"✓ Found {len(las_files)} LAS files")
    print(f"  Examples: {', '.join([f.name for f in las_files[:3]])}")
    
    return True


def check_pretrained(cfg):
    """Check if pretrained checkpoint exists (optional)."""
    print("\n" + "="*60)
    print("Checking Pretrained Checkpoint (Optional)")
    print("="*60)
    
    if cfg is None:
        print("⚠ Cannot check pretrained checkpoint (config not loaded)")
        return True  # Not critical
    
    if not hasattr(cfg.model, 'pretrained_encoder_path') or not cfg.model.pretrained_encoder_path:
        print("⚠ No pretrained checkpoint specified (training from scratch)")
        return True
    
    pretrained_path = Path(cfg.model.pretrained_encoder_path)
    
    if not pretrained_path.exists():
        print(f"⚠ Pretrained checkpoint not found: {pretrained_path}")
        print("  → Training will start from scratch")
        print("  → Update 'model.pretrained_encoder_path' to use pretrained weights")
        return True  # Not critical, can train from scratch
    
    print(f"✓ Pretrained checkpoint exists: {pretrained_path}")
    
    try:
        import torch
        checkpoint = torch.load(str(pretrained_path), map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        print(f"✓ Checkpoint contains {len(state_dict)} parameters")
        
    except Exception as e:
        print(f"⚠ Could not load checkpoint: {e}")
        return True  # Not critical
    
    return True


def check_gpu():
    """Check GPU availability."""
    print("\n" + "="*60)
    print("Checking GPU")
    print("="*60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠ CUDA not available (will use CPU - slower)")
            return True  # Not critical
    except Exception as e:
        print(f"⚠ Could not check GPU: {e}")
        return True


def main():
    parser = argparse.ArgumentParser(description='Verify HarvardForest3D SSL setup')
    parser.add_argument(
        '--cfg',
        type=str,
        default='ml3d/configs/harvardforest3d_ssl.yml',
        help='Path to config file'
    )
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("HarvardForest3D SSL Setup Verification")
    print("="*60)
    
    # Run checks
    checks = []
    
    checks.append(("Packages", check_packages()))
    checks.append(("Modules", check_modules()))
    
    cfg_ok, cfg = check_config(args.cfg)
    checks.append(("Config", cfg_ok))
    
    checks.append(("Dataset", check_dataset(cfg)))
    checks.append(("Pretrained (optional)", check_pretrained(cfg)))
    checks.append(("GPU (optional)", check_gpu()))
    
    # Summary
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)
    
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:30s} {status}")
    
    critical_checks = checks[:4]  # Packages, Modules, Config, Dataset
    all_critical_passed = all(passed for _, passed in critical_checks)
    
    print("\n" + "="*60)
    
    if all_critical_passed:
        print("✓ Setup verification PASSED")
        print("\nYou can start training with:")
        print(f"  python scripts/train_harvardforest_ssl.py --cfg {args.cfg} --device cuda")
        print("\nOr test imports with:")
        print("  python -c 'from ml3d.datasets import HarvardForest3D; print(\"OK\")'")
        return 0
    else:
        print("✗ Setup verification FAILED")
        print("\nPlease fix the issues above before training.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
