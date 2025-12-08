#!/usr/bin/env python3
"""
Debug script to understand CORAL loss normalization.
Shows why CORAL loss appears tiny (e.g., 9.5e-08).
"""

import torch
import numpy as np

def compute_covariance(features):
    """Compute covariance matrix."""
    n = features.size(0)
    features_centered = features - features.mean(dim=0, keepdim=True)
    cov = torch.mm(features_centered.t(), features_centered) / (n - 1)
    return cov

def analyze_coral_loss_scales():
    """Analyze CORAL loss at different feature dimensions."""
    
    print("="*70)
    print("CORAL Loss Normalization Analysis")
    print("="*70)
    print("\nShowing why CORAL loss appears tiny (e.g., 9.5e-08)\n")
    
    # Simulate features with different dimensions (like RandLANet layers)
    dimensions = [16, 64, 128, 256, 512]  # RandLANet output dimensions
    n_samples = 1000
    
    for d in dimensions:
        print(f"\n{'='*70}")
        print(f"Feature Dimension D = {d}")
        print(f"{'='*70}")
        
        # Generate random features (simulating source and target)
        # Using slightly different distributions to simulate domain gap
        source_features = torch.randn(n_samples, d) * 1.0 + 0.0
        target_features = torch.randn(n_samples, d) * 1.2 + 0.3  # Different scale/shift
        
        # Compute covariance matrices
        source_cov = compute_covariance(source_features)
        target_cov = compute_covariance(target_features)
        
        # Compute Frobenius norm of difference
        cov_diff = source_cov - target_cov
        frobenius_norm = torch.norm(cov_diff, p='fro')
        frobenius_squared = frobenius_norm ** 2
        
        # Different normalization strategies
        norm_4d2 = 4 * d * d  # Current implementation
        norm_d2 = d * d
        norm_d = d
        norm_none = 1
        
        loss_4d2 = frobenius_squared / norm_4d2
        loss_d2 = frobenius_squared / norm_d2
        loss_d = frobenius_squared / norm_d
        loss_none = frobenius_squared
        
        print(f"\nRaw Frobenius norm: {frobenius_norm:.4f}")
        print(f"Raw Frobenius squared: {frobenius_squared:.4f}")
        print(f"\nNormalization comparison:")
        print(f"  No normalization (÷1):     {loss_none:.6f}")
        print(f"  Divide by D (÷{d}):        {loss_d:.6f}")
        print(f"  Divide by D² (÷{d*d}):     {loss_d2:.8f}")
        print(f"  Divide by 4D² (÷{norm_4d2}): {loss_4d2:.10f}  ← CURRENT")
        
        print(f"\n  Impact of 4D² normalization:")
        print(f"    D={d} → dividing by {norm_4d2:,}")
        print(f"    Makes loss {norm_4d2:,}x smaller!")
        
        # Show what happens with typical covariance values
        cov_trace_s = source_cov.trace().item()
        cov_trace_t = target_cov.trace().item()
        print(f"\n  Covariance statistics:")
        print(f"    Source cov trace: {cov_trace_s:.4f}")
        print(f"    Target cov trace: {cov_trace_t:.4f}")
        print(f"    Cov difference magnitude: {frobenius_norm:.4f}")

    # Final recommendations
    print(f"\n{'='*70}")
    print("CONCLUSIONS:")
    print("="*70)
    print("""
1. Current normalization (÷4D²) makes loss EXTREMELY small for large D
   - D=16:  divides by 1,024
   - D=512: divides by 1,048,576 (1 million!)
   
2. This explains your tiny loss: 9.5e-08
   - Raw loss might be ~100-1000
   - After ÷4D² → becomes 1e-07 or smaller
   
3. Why this matters:
   - Loss magnitude affects gradient flow
   - Very small losses → very small gradients
   - May need to increase coral_weight significantly
   
4. Recommended fixes:
   a) Increase coral_weight from 0.1 to 1.0-10.0
   b) Or change normalization to ÷D or ÷√D instead of ÷4D²
   c) Or remove the factor of 4 (use ÷D² instead)
   
5. Original CORAL paper uses ÷4D² for theoretical reasons:
   - Makes loss dimension-independent
   - But may be too aggressive for deep learning
   
6. SqueezeSegV2 (your reference) likely compensates with higher weight
    """)
    
    print("\n" + "="*70)
    print("Recommendation for your setup:")
    print("="*70)
    print(f"""
With alignment_layers: [0, 2, 4] having dimensions ~[64, 128, 512]:
- Average normalization factor: ~200,000
- Current coral_weight: 0.1
- Effective weight: 0.1 / 200,000 = 5e-07

Try one of:
1. Keep normalization, increase coral_weight to 10.0 or 100.0
2. Change normalization factor in coral_loss.py line 71:
   From: loss = loss / (4 * d * d)
   To:   loss = loss / d  # or loss = loss / (d * d)
    """)

if __name__ == "__main__":
    analyze_coral_loss_scales()
