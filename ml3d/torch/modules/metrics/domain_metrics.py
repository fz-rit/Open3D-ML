"""
Domain Adaptation Metrics for monitoring feature alignment quality.
Includes MMD, covariance distance, and distribution statistics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_mmd(source_features: torch.Tensor, 
                target_features: torch.Tensor,
                kernel: str = 'rbf',
                sigma: Optional[float] = None,
                max_samples: int = 5000) -> float:
    """
    Compute Maximum Mean Discrepancy between source and target features.
    
    Args:
        source_features: [N_s, D] source domain features
        target_features: [N_t, D] target domain features
        kernel: 'rbf' or 'linear'
        sigma: RBF kernel bandwidth (auto-computed if None)
        max_samples: Maximum samples per domain to prevent O(N²) memory explosion
    
    Returns:
        MMD distance (scalar)
    """
    # Subsample if too many features (prevents 38TB allocation!)
    if source_features.size(0) > max_samples:
        indices = torch.randperm(source_features.size(0), device=source_features.device)[:max_samples]
        source_features = source_features[indices]
    
    if target_features.size(0) > max_samples:
        indices = torch.randperm(target_features.size(0), device=target_features.device)[:max_samples]
        target_features = target_features[indices]
    
    n_s = source_features.size(0)
    n_t = target_features.size(0)
    
    if kernel == 'rbf':
        if sigma is None:
            # Median heuristic for bandwidth
            all_features = torch.cat([source_features, target_features], dim=0)
            dists = torch.cdist(all_features, all_features)
            sigma = torch.median(dists[dists > 0])
        
        def rbf_kernel(x, y):
            dists = torch.cdist(x, y)
            return torch.exp(-dists ** 2 / (2 * sigma ** 2))
        
        K_ss = rbf_kernel(source_features, source_features)
        K_tt = rbf_kernel(target_features, target_features)
        K_st = rbf_kernel(source_features, target_features)
        
    else:  # linear kernel
        K_ss = torch.mm(source_features, source_features.t())
        K_tt = torch.mm(target_features, target_features.t())
        K_st = torch.mm(source_features, target_features.t())
    
    # MMD^2 = E[K(s,s)] + E[K(t,t)] - 2*E[K(s,t)]
    mmd = (K_ss.sum() - K_ss.trace()) / (n_s * (n_s - 1))
    mmd += (K_tt.sum() - K_tt.trace()) / (n_t * (n_t - 1))
    mmd -= 2 * K_st.sum() / (n_s * n_t)
    
    return mmd.item()


def compute_covariance_distance(source_features: torch.Tensor,
                                target_features: torch.Tensor,
                                use_geodesic: bool = True) -> float:
    """
    Compute distance between covariance matrices (CORAL metric).
    
    Args:
        source_features: [N_s, D] source features
        target_features: [N_t, D] target features
        use_geodesic: Use geodesic distance on SPD manifold
    
    Returns:
        Covariance distance (scalar)
    """
    # Center features
    source_centered = source_features - source_features.mean(dim=0, keepdim=True)
    target_centered = target_features - target_features.mean(dim=0, keepdim=True)
    
    # Compute covariances
    n_s = source_features.size(0)
    n_t = target_features.size(0)
    d = source_features.size(1)
    
    cov_s = torch.mm(source_centered.t(), source_centered) / (n_s - 1)
    cov_t = torch.mm(target_centered.t(), target_centered) / (n_t - 1)
    
    # Add regularization for numerical stability
    cov_s += torch.eye(d, device=cov_s.device) * 1e-5
    cov_t += torch.eye(d, device=cov_t.device) * 1e-5
    
    if use_geodesic:
        try:
            # Geodesic distance: ||log(Cs^{-1/2} Ct Cs^{-1/2})||_F
            cov_s_sqrt_inv = torch.linalg.cholesky(cov_s).inverse()
            M = torch.mm(torch.mm(cov_s_sqrt_inv.t(), cov_t), cov_s_sqrt_inv)
            eigvals = torch.linalg.eigvalsh(M)
            eigvals = torch.clamp(eigvals, min=1e-7)
            distance = torch.sqrt(torch.sum(torch.log(eigvals) ** 2)) / (2 * d)
        except:
            # Fallback to Frobenius norm
            distance = torch.norm(cov_s - cov_t, p='fro') / (2 * d * d)
    else:
        # Frobenius norm distance
        distance = torch.norm(cov_s - cov_t, p='fro') / (2 * d * d)
    
    return distance.item()


def compute_feature_statistics(features: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistical properties of features.
    
    Args:
        features: [N, D] feature tensor
    
    Returns:
        Dictionary with mean, std, min, max, sparsity
    """
    stats = {
        'mean': features.mean().item(),
        'std': features.std().item(),
        'min': features.min().item(),
        'max': features.max().item(),
        'l2_norm_mean': torch.norm(features, dim=1).mean().item(),
        'sparsity': (features.abs() < 1e-5).float().mean().item(),
    }
    return stats


def compute_class_wise_metrics(features: torch.Tensor,
                               labels: torch.Tensor,
                               num_classes: int) -> Dict[int, Dict[str, float]]:
    """
    Compute feature statistics per class.
    
    Args:
        features: [N, D] feature tensor
        labels: [N] class labels
        num_classes: Number of classes
    
    Returns:
        Dictionary mapping class_id -> statistics
    """
    class_stats = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_features = features[mask]
            class_stats[c] = compute_feature_statistics(class_features)
            class_stats[c]['count'] = mask.sum().item()
        else:
            class_stats[c] = {'count': 0}
    
    return class_stats


def compute_layer_alignment_quality(source_features_list: List[torch.Tensor],
                                    target_features_list: List[torch.Tensor],
                                    use_geodesic: bool = True) -> Dict[int, Dict[str, float]]:
    """
    Compute alignment metrics for each layer.
    
    Args:
        source_features_list: List of [N_s, D_i] source features per layer
        target_features_list: List of [N_t, D_i] target features per layer
        use_geodesic: Use geodesic distance for covariance
    
    Returns:
        Dictionary mapping layer_idx -> {mmd, cov_dist, ...}
    """
    layer_metrics = {}
    
    for i, (source_feat, target_feat) in enumerate(zip(source_features_list, target_features_list)):
        # Ensure same dimension
        if source_feat.size(1) != target_feat.size(1):
            continue
            
        metrics = {
            'mmd_rbf': compute_mmd(source_feat, target_feat, kernel='rbf'),
            'mmd_linear': compute_mmd(source_feat, target_feat, kernel='linear'),
            'cov_distance': compute_covariance_distance(source_feat, target_feat, use_geodesic),
            'source_stats': compute_feature_statistics(source_feat),
            'target_stats': compute_feature_statistics(target_feat),
        }
        
        layer_metrics[i] = metrics
    
    return layer_metrics


def compute_domain_gap_reduction(baseline_mmd: float,
                                 current_mmd: float) -> float:
    """
    Compute percentage of domain gap reduced.
    
    Args:
        baseline_mmd: MMD before domain adaptation
        current_mmd: MMD after domain adaptation
    
    Returns:
        Gap reduction percentage (0-100)
    """
    if baseline_mmd <= 0:
        return 0.0
    reduction = (baseline_mmd - current_mmd) / baseline_mmd * 100
    return max(0.0, reduction)


def compute_a_distance(source_features: torch.Tensor,
                       target_features: torch.Tensor,
                       max_samples: int = 5000) -> float:
    """
    Compute A-distance (proxy) between domains using linear classifier.
    Lower A-distance indicates better domain alignment.
    
    Args:
        source_features: [N_s, D] source features
        target_features: [N_t, D] target features
        max_samples: Maximum samples per domain for efficient training
    
    Returns:
        A-distance approximation
    """
    from sklearn.linear_model import LogisticRegression
    
    # Subsample if too many features
    if source_features.size(0) > max_samples:
        indices = torch.randperm(source_features.size(0))[:max_samples]
        source_features = source_features[indices]
    
    if target_features.size(0) > max_samples:
        indices = torch.randperm(target_features.size(0))[:max_samples]
        target_features = target_features[indices]
    
    # Create domain labels
    X = torch.cat([source_features, target_features], dim=0).cpu().numpy()
    y = np.concatenate([
        np.zeros(source_features.size(0)),
        np.ones(target_features.size(0))
    ])
    
    # Train domain classifier
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X, y)
    
    # A-distance ≈ 2(1 - 2*error)
    accuracy = clf.score(X, y)
    error = 1 - accuracy
    a_dist = 2 * (1 - 2 * error)
    
    return max(0.0, a_dist)


class DomainMetricsTracker:
    """
    Track domain adaptation metrics over training.
    """
    
    def __init__(self):
        self.history = {
            'epoch': [],
            'mmd': [],
            'cov_distance': [],
            'a_distance': [],
            'source_val_iou': [],
            'target_val_iou': [],
        }
        self.layer_history = {}  # layer_idx -> {metric: [values]}
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """Add metrics for current epoch."""
        self.history['epoch'].append(epoch)
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def update_layer_metrics(self, epoch: int, layer_metrics: Dict[int, Dict[str, float]]):
        """Add per-layer metrics."""
        for layer_idx, metrics in layer_metrics.items():
            if layer_idx not in self.layer_history:
                self.layer_history[layer_idx] = {'epoch': [], 'mmd_rbf': [], 'cov_distance': []}
            
            self.layer_history[layer_idx]['epoch'].append(epoch)
            self.layer_history[layer_idx]['mmd_rbf'].append(metrics['mmd_rbf'])
            self.layer_history[layer_idx]['cov_distance'].append(metrics['cov_distance'])
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if len(self.history['epoch']) == 0:
            return {}
        
        summary = {
            'total_epochs': len(self.history['epoch']),
            'final_mmd': self.history['mmd'][-1] if self.history['mmd'] else None,
            'mmd_reduction': (self.history['mmd'][0] - self.history['mmd'][-1]) if len(self.history['mmd']) > 1 else 0,
            'best_target_iou': max(self.history['target_val_iou']) if self.history['target_val_iou'] else None,
        }
        return summary
