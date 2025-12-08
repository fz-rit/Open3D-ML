"""
CORAL (Correlation Alignment) Loss for Domain Adaptation.

Reference: 
- Deep CORAL: Correlation Alignment for Deep Domain Adaptation (2016)
- SqueezeSegV2: Improved Model Structure and Unsupervised Domain Adaptation 
  for Road-Object Segmentation from a LiDAR Point Cloud (arXiv:1809.08495)
"""

import torch
import torch.nn as nn
import logging

log = logging.getLogger(__name__)


class CORALLoss(nn.Module):
    """
    Correlation Alignment (CORAL) Loss for Domain Adaptation.
    
    Computes the geodesic distance between covariance matrices of 
    source and target domain features to align their second-order statistics.
    
    This implementation supports both standard Frobenius norm distance and
    geodesic distance on the manifold of symmetric positive definite matrices.
    
    Reference: SqueezeSegV2 (arXiv:1809.08495)
    """
    
    def __init__(self, use_geodesic=True):
        """
        Args:
            use_geodesic: If True, uses geodesic distance (more theoretically sound).
                         If False, uses Frobenius norm (faster, simpler).
        """
        super(CORALLoss, self).__init__()
        self.use_geodesic = use_geodesic
    
    def forward(self, source_features, target_features, debug=False):
        """
        Compute CORAL loss between source and target domain features.
        
        Args:
            source_features: (N_s, D) tensor of source domain features
            target_features: (N_t, D) tensor of target domain features
            debug: If True, print intermediate values for debugging
            
        Returns:
            coral_loss: scalar tensor representing domain distance
        """
        # Ensure features are 2D
        if source_features.dim() > 2:
            source_features = source_features.view(-1, source_features.size(-1))
        if target_features.dim() > 2:
            target_features = target_features.view(-1, target_features.size(-1))
        
        # Skip if either domain has too few samples
        if source_features.size(0) < 2 or target_features.size(0) < 2:
            return torch.tensor(0.0, device=source_features.device, requires_grad=True)
        
        if debug:
            log.info(f"\n=== CORAL Loss Debug ===")
            log.info(f"Source features shape: {source_features.shape}")
            log.info(f"Target features shape: {target_features.shape}")
            log.info(f"Source features range: [{source_features.min():.6f}, {source_features.max():.6f}]")
            log.info(f"Target features range: [{target_features.min():.6f}, {target_features.max():.6f}]")
            log.info(f"Source features mean: {source_features.mean():.6f}, std: {source_features.std():.6f}")
            log.info(f"Target features mean: {target_features.mean():.6f}, std: {target_features.std():.6f}")
        
        # Compute covariance matrices
        source_cov = self._compute_covariance(source_features)
        target_cov = self._compute_covariance(target_features)
        
        if debug:
            log.info(f"\nCovariance matrices:")
            log.info(f"Source cov shape: {source_cov.shape}")
            log.info(f"Source cov diagonal mean: {source_cov.diag().mean():.6f}")
            log.info(f"Source cov Frobenius norm: {torch.norm(source_cov, p='fro'):.6f}")
            log.info(f"Target cov diagonal mean: {target_cov.diag().mean():.6f}")
            log.info(f"Target cov Frobenius norm: {torch.norm(target_cov, p='fro'):.6f}")
            cov_diff = source_cov - target_cov
            log.info(f"Cov difference Frobenius norm: {torch.norm(cov_diff, p='fro'):.6f}")
        
        # Compute distance between covariance matrices
        if self.use_geodesic:
            loss = self._geodesic_distance(source_cov, target_cov, debug=debug)
        else:
            # Frobenius norm (standard CORAL)
            loss = torch.norm(source_cov - target_cov, p='fro') ** 2
            if debug:
                log.info(f"\nFrobenius loss (before normalization): {loss:.6f}")
        
        # Normalize by dimension
        d = source_features.size(1)
        loss_before_norm = loss.item() if isinstance(loss, torch.Tensor) else loss
        loss = loss / (4 * d * d)
        
        # Protect against NaN/Inf from unstable geodesic computation
        if torch.isnan(loss) or torch.isinf(loss):
            if debug:
                log.warning(f"NaN/Inf detected in CORAL loss. Replacing with 0.0")
            loss = torch.tensor(0.0, device=source_features.device, requires_grad=True)
        
        if debug:
            log.info(f"\nNormalization:")
            log.info(f"Feature dimension D: {d}")
            log.info(f"Normalization factor (4*D²): {4 * d * d}")
            log.info(f"Loss before normalization: {loss_before_norm:.6f}")
            log.info(f"Loss after normalization: {loss:.10f}")
            log.info(f"======================\n")
        
        return loss
    
    def _compute_covariance(self, features):
        """
        Compute covariance matrix of features.
        
        Args:
            features: (N, D) tensor
            
        Returns:
            cov: (D, D) covariance matrix
        """
        n = features.size(0)
        
        # Center the features (zero mean)
        features_centered = features - features.mean(dim=0, keepdim=True)
        
        # Compute covariance: Cov = (X^T X) / (n - 1)
        cov = torch.mm(features_centered.t(), features_centered) / (n - 1)
        
        return cov
    
    def _geodesic_distance(self, A, B, debug=False):
        """
        Compute geodesic distance between two covariance matrices.
        
        For symmetric positive definite (SPD) matrices, the geodesic distance 
        on the Riemannian manifold is:
            d(A, B) = ||log(A^{-1/2} B A^{-1/2})||_F
        
        This is more theoretically sound than Frobenius norm as it respects
        the geometry of the SPD manifold.
        
        Args:
            A, B: (D, D) covariance matrices
            debug: If True, print intermediate values
            
        Returns:
            distance: scalar tensor
        """
        # Add small epsilon for numerical stability
        eps = 1e-5
        d = A.size(0)
        A = A + eps * torch.eye(d, device=A.device, dtype=A.dtype)
        B = B + eps * torch.eye(d, device=B.device, dtype=B.dtype)
        
        try:
            # Compute A^{-1/2} using eigenvalue decomposition
            # A = Q Λ Q^T, so A^{-1/2} = Q Λ^{-1/2} Q^T
            eigvals_A, eigvecs_A = torch.linalg.eigh(A)
            eigvals_A = torch.clamp(eigvals_A, min=eps)
            A_inv_sqrt = eigvecs_A @ torch.diag(1.0 / torch.sqrt(eigvals_A)) @ eigvecs_A.t()
            
            # Compute M = A^{-1/2} B A^{-1/2}
            M = A_inv_sqrt @ B @ A_inv_sqrt
            
            # Eigenvalue decomposition for M
            eigvals_M, _ = torch.linalg.eigh(M)
            eigvals_M = torch.clamp(eigvals_M, min=eps)
            
            if debug:
                log.info(f"\nGeodesic distance computation:")
                log.info(f"A eigenvalues range: [{eigvals_A.min():.6f}, {eigvals_A.max():.6f}]")
                log.info(f"M eigenvalues range: [{eigvals_M.min():.6f}, {eigvals_M.max():.6f}]")
                log.info(f"log(M eigenvalues) range: [{torch.log(eigvals_M).min():.6f}, {torch.log(eigvals_M).max():.6f}]")
            
            # Compute ||log(M)||_F = sqrt(sum(log(eigvals)^2))
            log_eigvals = torch.log(eigvals_M)
            distance = torch.sqrt(torch.sum(log_eigvals ** 2))
            
            if debug:
                log.info(f"Geodesic distance (before normalization): {distance:.6f}")
            
        except Exception as e:
            # Fallback to Frobenius norm if eigendecomposition fails
            # This can happen with ill-conditioned matrices
            distance = torch.norm(A - B, p='fro')
            if debug:
                log.info(f"Geodesic failed, using Frobenius fallback: {distance:.6f}")
                log.info(f"Error: {e}")
        
        return distance


class MultiLayerCORALLoss(nn.Module):
    """
    Multi-layer CORAL loss that aligns features at multiple network depths.
    
    This is more effective than single-layer alignment as it aligns 
    representations at different semantic levels (low-level features to 
    high-level features).
    
    Usage:
        # During training, collect features from multiple layers
        source_feats = [encoder1_out, encoder2_out, encoder3_out]
        target_feats = [encoder1_out, encoder2_out, encoder3_out]
        coral_loss = multi_coral(source_feats, target_feats)
    """
    
    def __init__(self, layer_weights=None, use_geodesic=True):
        """
        Args:
            layer_weights: List of weights for each layer's CORAL loss.
                          If None, equal weights are used.
            use_geodesic: Whether to use geodesic distance (passed to CORALLoss).
        """
        super(MultiLayerCORALLoss, self).__init__()
        self.coral_loss = CORALLoss(use_geodesic=use_geodesic)
        self.layer_weights = layer_weights
    
    def forward(self, source_features_list, target_features_list, debug=False):
        """
        Compute weighted sum of CORAL losses across multiple layers.
        
        Args:
            source_features_list: List of (N_s, D_i) source feature tensors
            target_features_list: List of (N_t, D_i) target feature tensors
            debug: If True, print debug info for each layer
            
        Returns:
            total_loss: weighted sum of CORAL losses across layers
        """
        if len(source_features_list) != len(target_features_list):
            raise ValueError(
                f"Number of source layers ({len(source_features_list)}) must match "
                f"number of target layers ({len(target_features_list)})"
            )
        
        num_layers = len(source_features_list)
        
        # Default to equal weights if not specified
        if self.layer_weights is None:
            self.layer_weights = [1.0 / num_layers] * num_layers
        
        if len(self.layer_weights) != num_layers:
            raise ValueError(
                f"Number of layer weights ({len(self.layer_weights)}) must match "
                f"number of layers ({num_layers})"
            )
        
        if debug:
            log.info(f"\n{'='*60}")
            log.info(f"MultiLayer CORAL Loss Debug (total {num_layers} layers)")
            log.info(f"Layer weights: {self.layer_weights}")
            log.info(f"{'='*60}")
        
        total_loss = 0
        for i, (src_feat, tgt_feat) in enumerate(zip(source_features_list, 
                                                       target_features_list)):
            if debug:
                log.info(f"\n--- Layer {i} ---")
            layer_loss = self.coral_loss(src_feat, tgt_feat, debug=debug)
            weighted_loss = self.layer_weights[i] * layer_loss
            if debug:
                log.info(f"Layer {i} loss: {layer_loss:.10f}")
                log.info(f"Layer {i} weight: {self.layer_weights[i]}")
                log.info(f"Layer {i} weighted loss: {weighted_loss:.10f}")
            total_loss += weighted_loss
        
        if debug:
            log.info(f"\n{'='*60}")
            log.info(f"Total MultiLayer CORAL Loss: {total_loss:.10f}")
            log.info(f"{'='*60}\n")
        
        return total_loss


class AdaptiveCORALLoss(nn.Module):
    """
    Adaptive CORAL loss with progressive domain calibration.
    
    Gradually increases the weight of domain adaptation loss during training,
    allowing the model to first learn task-specific features before aligning domains.
    
    This implements the "progressive domain calibration" strategy from SqueezeSegV2.
    """
    
    def __init__(self, 
                 base_weight=1.0, 
                 ramp_up_steps=5000,
                 use_geodesic=True,
                 min_weight_ratio=0.1):
        """
        Args:
            base_weight: Initial weight for CORAL loss (starts at this value).
            ramp_up_steps: Number of training steps to ramp down from base_weight to min_weight.
            use_geodesic: Whether to use geodesic distance.
            min_weight_ratio: Minimum weight as ratio of base_weight (default: 0.1).
                             Final weight = base_weight * min_weight_ratio.
        """
        super(AdaptiveCORALLoss, self).__init__()
        self.base_weight = base_weight
        self.ramp_up_steps = ramp_up_steps
        self.min_weight_ratio = min_weight_ratio
        self.coral_loss = CORALLoss(use_geodesic=use_geodesic)
        self.global_step = 0
    
    def forward(self, source_features, target_features):
        """
        Compute CORAL loss with adaptive weight based on training progress.
        
        Args:
            source_features: Source domain features
            target_features: Target domain features
            
        Returns:
            weighted_loss: CORAL loss multiplied by adaptive weight
        """
        # Compute base CORAL loss
        coral_loss = self.coral_loss(source_features, target_features)
        
        # Compute adaptive weight (inverted: start high, ramp down)
        if self.ramp_up_steps > 0:
            # Progress from 0 (start) to 1 (end of ramp)
            progress = min(1.0, self.global_step / self.ramp_up_steps)
            # Weight starts at base_weight, decreases to base_weight * min_weight_ratio
            adaptive_weight = self.base_weight * (1.0 - progress * (1.0 - self.min_weight_ratio))
        else:
            adaptive_weight = self.base_weight
        
        # Return weighted loss
        return adaptive_weight * coral_loss
    
    def step(self):
        """Increment the global step counter. Call this after each training iteration."""
        self.global_step += 1
    
    def reset_step(self):
        """Reset the global step counter."""
        self.global_step = 0
    
    def get_current_weight(self):
        """Get the current adaptive weight value."""
        if self.ramp_up_steps > 0:
            progress = min(1.0, self.global_step / self.ramp_up_steps)
            return self.base_weight * (1.0 - progress * (1.0 - self.min_weight_ratio))
        else:
            return self.base_weight


# Alias for convenience
CorrelationAlignmentLoss = CORALLoss
