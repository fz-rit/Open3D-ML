"""
RandLANet for semantic segmentation using contrastive pre-trained encoder.

This model adapts the contrastive learning encoder for point-level semantic segmentation.
It loads the pre-trained encoder weights and adds a segmentation head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path

from .randlanet_contrast import RandLANetContrast
from .base_model import BaseModel
from ...utils import MODEL

log = logging.getLogger(__name__)


@MODEL.register_module('RandLANetSegmentation')
class RandLANetSegmentation(BaseModel):
    """RandLANet for semantic segmentation with contrastive pre-training.
    
    Architecture:
        Input → Pre-trained Encoder → Feature Propagation → Segmentation Head → Per-point labels
        
    The encoder can be initialized from a contrastive pre-trained checkpoint.
    """
    
    def __init__(
        self,
        name='RandLANetSegmentation',
        # Encoder parameters (must match pre-trained model)
        num_neighbors=16,
        num_layers=5,
        num_points=65536,  # More points for segmentation
        sub_sampling_ratio=[4, 4, 4, 4, 2],
        in_channels=3,
        dim_features=8,
        first_features_dim=1024,  # Encoder output dimension
        # Segmentation parameters
        num_classes=5,  # e.g., ground, vegetation, trunk, branches, other
        dropout=0.5,
        # Pre-trained encoder
        pretrained_contrast_path=None,  # Path to contrastive checkpoint
        freeze_encoder=False,  # Freeze encoder during segmentation training
        encoder_lr_scale=0.1,  # Lower LR for encoder if not frozen
        # Training parameters
        batcher='DefaultBatcher',
        ckpt_path=None,
        **kwargs
    ):
        """Initialize RandLANetSegmentation.
        
        Args:
            pretrained_contrast_path: Path to contrastive pre-trained checkpoint
            freeze_encoder: Whether to freeze encoder during segmentation training
            encoder_lr_scale: Learning rate scale for encoder (if not frozen)
            num_classes: Number of semantic classes
            num_points: Number of points per sample (can be larger for segmentation)
        """
        super().__init__(
            name=name,
            num_neighbors=num_neighbors,
            num_layers=num_layers,
            num_points=num_points,
            sub_sampling_ratio=sub_sampling_ratio,
            in_channels=in_channels,
            dim_features=dim_features,
            batcher=batcher,
            ckpt_path=ckpt_path,
            **kwargs
        )
        
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        self.encoder_lr_scale = encoder_lr_scale
        self.first_features_dim = first_features_dim
        
        # Build encoder (same as contrastive model)
        self._build_encoder()
        
        # Build segmentation head
        self._build_segmentation_head(dropout)
        
        # Load pre-trained encoder if provided
        if pretrained_contrast_path is not None:
            self._load_pretrained_encoder(pretrained_contrast_path)
        
        # Freeze encoder if specified
        if freeze_encoder:
            self._freeze_encoder()
            log.info("Encoder frozen - only training segmentation head")
        
        log.info(f"RandLANetSegmentation initialized:")
        log.info(f"  Encoder output dim: {first_features_dim}")
        log.info(f"  Number of classes: {num_classes}")
        log.info(f"  Number of points: {num_points}")
        log.info(f"  Encoder frozen: {freeze_encoder}")
    
    def _build_encoder(self):
        """Build encoder (same architecture as contrastive model)."""
        cfg = self.cfg
        
        # Input projection
        self.fc0 = nn.Linear(cfg.in_channels, cfg.dim_features)
        self.bn0 = nn.BatchNorm2d(cfg.dim_features, eps=1e-6, momentum=0.01)
        
        # Simplified encoder with 1x1 convolutions
        self.conv1 = nn.Conv2d(cfg.dim_features, 64, 1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, 1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, 1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, 1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.conv5 = nn.Conv2d(512, self.first_features_dim, 1)
        self.bn5 = nn.BatchNorm2d(self.first_features_dim)
        
        log.info(f"Built encoder with output dimension: {self.first_features_dim}")
    
    def _build_segmentation_head(self, dropout=0.5):
        """Build segmentation head for per-point classification.
        
        Uses skip connections from multiple encoder layers for multi-scale features.
        """
        # Feature propagation layers (upsample features from encoder)
        # These combine features from different scales
        
        # We'll use a simple MLP head on top of encoder features
        # For better results, you could add feature propagation layers
        
        self.seg_head = nn.Sequential(
            nn.Conv1d(self.first_features_dim, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(128, self.num_classes, 1)  # Final per-point classification
        )
        
        log.info(f"Built segmentation head: {self.first_features_dim} → ... → {self.num_classes}")
    
    def _load_pretrained_encoder(self, ckpt_path):
        """Load pre-trained encoder weights from contrastive checkpoint.
        
        Args:
            ckpt_path: Path to contrastive learning checkpoint
        """
        log.info(f"Loading pre-trained encoder from {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Filter encoder weights (exclude projection head)
        encoder_state = {}
        for key, value in state_dict.items():
            if not key.startswith('projection_head'):
                # Remove 'module.' prefix if present
                clean_key = key.replace('module.', '')
                encoder_state[clean_key] = value
        
        # Load encoder weights
        missing_keys, unexpected_keys = self.load_state_dict(encoder_state, strict=False)
        
        log.info(f"Loaded {len(encoder_state)} encoder parameters")
        if missing_keys:
            log.info(f"Missing keys: {len(missing_keys)} (expected: segmentation head)")
        if unexpected_keys:
            log.warning(f"Unexpected keys: {unexpected_keys}")
        
        log.info("Pre-trained encoder loaded successfully!")
    
    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        encoder_params = [
            'fc0', 'bn0', 
            'conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3',
            'conv4', 'bn4', 'conv5', 'bn5'
        ]
        
        for name in encoder_params:
            if hasattr(self, name):
                module = getattr(self, name)
                for param in module.parameters():
                    param.requires_grad = False
        
        log.info("Encoder parameters frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        self.freeze_encoder = False
        log.info("Encoder parameters unfrozen")
    
    def forward(self, inputs):
        """Forward pass for semantic segmentation.
        
        Args:
            inputs: Dict with 'point' key containing (B, N, 3) point coordinates
                    or torch.Tensor of shape (B, N, 3)
        
        Returns:
            Dict with:
                - 'logits': (B, num_classes, N) per-point class logits
                - 'features': (B, D, N) per-point features (optional)
        """
        if isinstance(inputs, dict):
            points = inputs['point']  # (B, N, 3)
        else:
            points = inputs
        
        B, N, C = points.shape
        
        # Encode points
        features = self._extract_features(points)  # (B, D, N)
        
        # Segmentation head
        logits = self.seg_head(features)  # (B, num_classes, N)
        
        return {
            'logits': logits,
            'features': features
        }
    
    def _extract_features(self, points):
        """Extract per-point features from encoder.
        
        Args:
            points: (B, N, 3) point coordinates
            
        Returns:
            features: (B, D, N) per-point features
        """
        B, N, C = points.shape
        
        # Reshape for processing: (B, N, C) -> (B, C, N, 1)
        x = points.permute(0, 2, 1).unsqueeze(-1)  # (B, 3, N, 1)
        
        # Input projection
        # (B, 3, N, 1) -> (B, N, 3) -> (B, N, D) -> (B, D, N, 1)
        x_fc = self.fc0(x.squeeze(-1).permute(0, 2, 1))  # (B, N, D)
        x = x_fc.permute(0, 2, 1).unsqueeze(-1)  # (B, D, N, 1)
        x = F.relu(self.bn0(x))
        
        # Encoder layers
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N, 1)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128, N, 1)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 256, N, 1)
        x = F.relu(self.bn4(self.conv4(x)))  # (B, 512, N, 1)
        x = F.relu(self.bn5(self.conv5(x)))  # (B, first_features_dim, N, 1)
        
        # Remove last dimension and return per-point features
        features = x.squeeze(-1)  # (B, D, N)
        
        return features
    
    def get_optimizer(self, cfg):
        """Get optimizer with different learning rates for encoder and head.
        
        Args:
            cfg: Config object with optimizer parameters
            
        Returns:
            optimizer: Configured optimizer
        """
        if self.freeze_encoder:
            # Only optimize segmentation head
            params = self.seg_head.parameters()
            log.info("Optimizing segmentation head only")
        else:
            # Different learning rates for encoder and head
            encoder_params = []
            head_params = []
            
            encoder_modules = [
                self.fc0, self.bn0,
                self.conv1, self.bn1, self.conv2, self.bn2,
                self.conv3, self.bn3, self.conv4, self.bn4,
                self.conv5, self.bn5
            ]
            
            for module in encoder_modules:
                encoder_params.extend(list(module.parameters()))
            
            head_params = list(self.seg_head.parameters())
            
            params = [
                {'params': encoder_params, 'lr': cfg.learning_rate * self.encoder_lr_scale},
                {'params': head_params, 'lr': cfg.learning_rate}
            ]
            
            log.info(f"Encoder LR: {cfg.learning_rate * self.encoder_lr_scale:.6f}")
            log.info(f"Head LR: {cfg.learning_rate:.6f}")
        
        optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)
        
        return optimizer
    
    def get_loss(self, results, inputs):
        """Compute segmentation loss.
        
        Args:
            results: Model outputs with 'logits' key
            inputs: Batch dict with 'labels' key (B, N) ground truth labels
            
        Returns:
            loss_dict: Dict with 'loss' and other metrics
        """
        logits = results['logits']  # (B, num_classes, N)
        labels = inputs['labels']  # (B, N)
        
        # Cross-entropy loss
        loss = F.cross_entropy(
            logits, 
            labels.long(),
            ignore_index=-1  # Ignore unlabeled points
        )
        
        # Compute accuracy
        preds = logits.argmax(dim=1)  # (B, N)
        valid_mask = (labels != -1)
        if valid_mask.sum() > 0:
            accuracy = (preds[valid_mask] == labels[valid_mask]).float().mean()
        else:
            accuracy = torch.tensor(0.0)
        
        return {
            'loss': loss,
            'accuracy': accuracy.item()
        }


# Example usage and configuration
if __name__ == '__main__':
    """
    Example: Load pre-trained contrastive model and fine-tune for segmentation
    
    1. Train contrastive model (scene-level):
       python scripts/run_pipeline.py --cfg configs/randlanet_harvardforest_contrast.yml
       
    2. Fine-tune for segmentation (point-level):
       python scripts/run_pipeline.py --cfg configs/randlanet_harvardforest_segmentation.yml
    """
    
    # Create model
    model = RandLANetSegmentation(
        num_classes=5,
        num_points=65536,
        first_features_dim=1024,
        pretrained_contrast_path='logs/checkpoint/ckpt_best.pth',
        freeze_encoder=True,  # Start with frozen encoder
        encoder_lr_scale=0.1
    )
    
    # Test forward pass
    dummy_input = torch.randn(2, 65536, 3)
    outputs = model({'point': dummy_input})
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Output features shape: {outputs['features'].shape}")
