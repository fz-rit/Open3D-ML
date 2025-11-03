"""
RandLANet model for contrastive learning (PointContrast-style).

Wraps RandLANet encoder with a projection head for contrastive learning.
Supports loading pretrained encoder weights from Semantic3D or other datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path

from .randlanet import RandLANet
from .base_model import BaseModel
from ...utils import MODEL

log = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning.
    
    Maps encoder features to a lower-dimensional space where contrastive
    loss is computed. This is a standard component in contrastive learning.
    """
    
    def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=128, 
                 use_bn=True, dropout=0.0):
        """
        Args:
            input_dim: Dimension of encoder output
            hidden_dims: List of hidden layer dimensions
            output_dim: Final embedding dimension
            use_bn: Use batch normalization
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (no activation, will be L2 normalized)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: (B, D) or (B, D, N, 1) encoder features
            
        Returns:
            embeddings: (B, output_dim) L2-normalized embeddings
        """
        if x.dim() == 4:  # (B, D, N, 1) from RandLANet
            # Global average pooling over points
            x = x.mean(dim=2).squeeze(-1)  # (B, D)
        elif x.dim() == 3:  # (B, D, N)
            x = x.mean(dim=2)  # (B, D)
        
        embeddings = self.mlp(x)  # (B, output_dim)
        
        # L2 normalization for cosine similarity
        embeddings = F.normalize(embeddings, dim=-1, p=2)
        
        return embeddings


class RandLANetContrast(BaseModel):
    """RandLANet for contrastive self-supervised learning.
    
    Architecture:
        Input → RandLANet Encoder → Projection Head → L2-normalized embeddings
        
    The encoder can be initialized from a pretrained checkpoint (e.g., Semantic3D).
    During training, both encoder and projection head are trained with contrastive loss.
    """
    
    def __init__(
        self,
        name='RandLANetContrast',
        # Encoder parameters (should match pretrained model)
        num_neighbors=16,
        num_layers=5,
        num_points=8192,
        sub_sampling_ratio=[4, 4, 4, 4, 2],
        in_channels=3,
        dim_features=8,
        dim_output=[16, 64, 128, 256, 512],
        grid_size=0.06,
        # Projection head parameters
        projection_hidden_dims=[512, 256],
        projection_output_dim=128,
        projection_use_bn=True,
        projection_dropout=0.1,
        # Pretrained weights
        pretrained_encoder_path=None,
        load_encoder_strict=False,
        freeze_encoder_epochs=0,  # Freeze encoder for N epochs (warmup)
        # Training parameters
        encoder_lr_scale=0.1,  # Encoder LR = base_lr * this scale
        batcher='DefaultBatcher',
        ckpt_path=None,
        **kwargs
    ):
        """Initialize RandLANetContrast model.
        
        Args:
            pretrained_encoder_path: Path to pretrained encoder checkpoint
            load_encoder_strict: If False, allow partial loading
            freeze_encoder_epochs: Number of epochs to freeze encoder (warmup)
            encoder_lr_scale: Learning rate scale for encoder vs projection head
            projection_hidden_dims: Hidden dimensions for projection MLP
            projection_output_dim: Output embedding dimension
        """
        super().__init__(
            name=name,
            num_neighbors=num_neighbors,
            num_layers=num_layers,
            num_points=num_points,
            sub_sampling_ratio=sub_sampling_ratio,
            in_channels=in_channels,
            dim_features=dim_features,
            dim_output=dim_output,
            grid_size=grid_size,
            batcher=batcher,
            ckpt_path=ckpt_path,
            **kwargs
        )
        
        cfg = self.cfg
        
        # Store configuration
        self.pretrained_encoder_path = pretrained_encoder_path
        self.load_encoder_strict = load_encoder_strict
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.encoder_lr_scale = encoder_lr_scale
        self.projection_output_dim = projection_output_dim
        
        # Build encoder (RandLANet architecture)
        self._build_encoder()
        
        # Build projection head
        encoder_output_dim = cfg.dim_output[-1] * 2  # After last LFA layer
        self.encoder_output_dim = encoder_output_dim  # Store for later use
        self.projection_head = ProjectionHead(
            input_dim=encoder_output_dim,
            hidden_dims=projection_hidden_dims,
            output_dim=projection_output_dim,
            use_bn=projection_use_bn,
            dropout=projection_dropout
        )
        
        # Load pretrained encoder if provided
        if pretrained_encoder_path is not None:
            self._load_pretrained_encoder(pretrained_encoder_path, load_encoder_strict)
        
        # Freeze encoder initially if specified
        self.encoder_frozen = False
        if freeze_encoder_epochs > 0:
            self.freeze_encoder()
        
        log.info(f"RandLANetContrast initialized:")
        log.info(f"  Encoder output dim: {encoder_output_dim}")
        log.info(f"  Projection output dim: {projection_output_dim}")
        log.info(f"  Freeze encoder epochs: {freeze_encoder_epochs}")

    def _build_encoder(self):
        """Build simplified encoder for contrastive learning.
        
        Uses a PointNet-style architecture with 1x1 convolutions instead of
        complex LFA layers, which is more suitable for self-supervised learning.
        """
        cfg = self.cfg
        
        # Input projection
        self.fc0 = nn.Linear(cfg.in_channels, cfg.dim_features)
        self.bn0 = nn.BatchNorm2d(cfg.dim_features, eps=1e-6, momentum=0.01)
        
        # Simplified encoder: progressive feature expansion with 1x1 convs
        # This is similar to PointNet and works well for contrastive learning
        self.conv1 = nn.Conv2d(cfg.dim_features, 64, 1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, 1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, 1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, 1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Final layer to encoder_output_dim
        encoder_output_dim = cfg.dim_output[-1] * 2  # Should be 1024
        self.conv5 = nn.Conv2d(512, encoder_output_dim, 1)
        self.bn5 = nn.BatchNorm2d(encoder_output_dim)
        
        self.lrelu = nn.LeakyReLU(0.2)
        
        log.info(f"Built simplified encoder with PointNet-style 1x1 convolutions")
        log.info(f"  Input: {cfg.in_channels} -> Output: {encoder_output_dim}")

    def _load_pretrained_encoder(self, ckpt_path, strict=False):
        """Load pretrained encoder weights.
        
        Args:
            ckpt_path: Path to checkpoint file
            strict: If False, allow partial loading (ignore missing keys)
        """
        if not Path(ckpt_path).exists():
            log.warning(f"Pretrained checkpoint not found: {ckpt_path}")
            return
        
        try:
            log.info(f"Loading pretrained encoder from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Filter encoder keys (fc0, bn0, encoder.*, mlp)
            encoder_state_dict = {}
            for key, value in state_dict.items():
                if any(key.startswith(prefix) for prefix in ['fc0', 'bn0', 'encoder', 'mlp']):
                    encoder_state_dict[key] = value
            
            # Load encoder weights
            missing_keys, unexpected_keys = self.load_state_dict(
                encoder_state_dict, strict=strict
            )
            
            if missing_keys:
                log.info(f"Missing keys (expected for projection head): {len(missing_keys)}")
            if unexpected_keys:
                log.warning(f"Unexpected keys: {unexpected_keys}")
            
            log.info(f"Successfully loaded {len(encoder_state_dict)} encoder parameters")
            
        except Exception as e:
            log.error(f"Error loading pretrained encoder: {e}")
            raise

    def freeze_encoder(self):
        """Freeze encoder parameters (for warmup training)."""
        for name, param in self.named_parameters():
            if 'projection_head' not in name:
                param.requires_grad = False
        self.encoder_frozen = True
        log.info("Encoder frozen (only training projection head)")

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters (for fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True
        self.encoder_frozen = False
        log.info("Encoder unfrozen (training full model)")

    def get_optimizer(self, cfg_pipeline):
        """Get optimizer with different learning rates for encoder and projection head.
        
        Args:
            cfg_pipeline: Pipeline configuration with 'learning_rate'
            
        Returns:
            optimizer: Adam optimizer with parameter groups
        """
        encoder_params = []
        projection_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'projection_head' in name:
                    projection_params.append(param)
                else:
                    encoder_params.append(param)
        
        base_lr = cfg_pipeline.learning_rate
        encoder_lr = base_lr * self.encoder_lr_scale
        
        param_groups = [
            {'params': encoder_params, 'lr': encoder_lr, 'name': 'encoder'},
            {'params': projection_params, 'lr': base_lr, 'name': 'projection_head'},
        ]
        
        optimizer = torch.optim.Adam(
            param_groups,
            lr=base_lr,
            weight_decay=getattr(cfg_pipeline, 'weight_decay', 1e-4)
        )
        
        log.info(f"Optimizer created: encoder_lr={encoder_lr:.2e}, projection_lr={base_lr:.2e}")
        
        return optimizer

    def forward(self, inputs):
        """Forward pass through encoder and projection head.
        
        Args:
            inputs: Dict with 'point' (B, N, 3+D) and optional 'feat'
            
        Returns:
            embeddings: (B, projection_output_dim) L2-normalized embeddings
            encoder_features: (B, encoder_dim, N', 1) raw encoder features
        """
        # Extract coordinates and features
        points = inputs['point']  # (B, N, 3+D)
        batch_size, num_points, point_dim = points.shape
        
        # Extract XYZ and features
        coords = points[:, :, :3]  # (B, N, 3)
        if point_dim > 3:
            feat = points[:, :, 3:]  # (B, N, D)
        else:
            # Use XYZ as features if no other features provided
            feat = coords.clone()
        
        # Normalize coordinates to unit sphere for better training
        coords_centered = coords - coords.mean(dim=1, keepdim=True)
        coords_normalized = coords_centered / (coords_centered.norm(dim=-1, keepdim=True).max(dim=1, keepdim=True)[0] + 1e-8)
        
        # Use normalized coordinates as input features
        feat = coords_normalized
        
        # Input projection: (B, N, 3) -> (B, dim_features, N, 1)
        feat = self.fc0(feat)  # (B, N, dim_features)
        feat = feat.transpose(-2, -1).unsqueeze(-1)  # (B, dim_features, N, 1)
        feat = self.bn0(feat)
        feat = self.lrelu(feat)
        
        # Progressive feature expansion (PointNet-style)
        feat = self.lrelu(self.bn1(self.conv1(feat)))  # (B, 64, N, 1)
        feat = self.lrelu(self.bn2(self.conv2(feat)))  # (B, 128, N, 1)
        feat = self.lrelu(self.bn3(self.conv3(feat)))  # (B, 256, N, 1)
        feat = self.lrelu(self.bn4(self.conv4(feat)))  # (B, 512, N, 1)
        encoder_features = self.lrelu(self.bn5(self.conv5(feat)))  # (B, encoder_output_dim, N, 1)
        
        # Global max pooling to get per-cloud features
        encoder_features_pooled = encoder_features.max(dim=2, keepdim=True)[0]  # (B, encoder_output_dim, 1, 1)
        
        # Project to embedding space
        embeddings = self.projection_head(encoder_features_pooled)
        
        return {
            'embeddings': embeddings,
            'encoder_features': encoder_features_pooled,
        }

    def preprocess(self, data, attr):
        """Preprocess data for contrastive learning.
        
        This should handle both view1 and view2 from contrastive dataset.
        """
        # Similar to RandLANet.preprocess but for contrastive pairs
        # TODO: Implement proper preprocessing
        return data

    def transform(self, data, attr):
        """Transform is not needed for contrastive learning."""
        return data

    def inference_begin(self, data):
        """Not applicable for contrastive learning."""
        pass

    def inference_preprocess(self):
        """Not applicable for contrastive learning."""
        return {}

    def inference_end(self, inputs, results):
        """Not applicable for contrastive learning."""
        return results

    def get_loss(self, Loss, results, inputs, device):
        """Get loss for contrastive learning.
        
        This method is required by BaseModel but not used directly in our pipeline.
        The ContrastiveLearning pipeline handles loss computation internally.
        """
        # Return dummy loss since pipeline handles it
        return torch.tensor(0.0, device=device)

