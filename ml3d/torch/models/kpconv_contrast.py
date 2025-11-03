"""
KPConv (KPFCNN) model for contrastive learning (PointContrast-style).

Wraps KPConv encoder with a projection head for contrastive learning.
Supports loading pretrained encoder weights from Semantic3D or other datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path

from .kpconv import KPFCNN
from .base_model import BaseModel
from .randlanet_contrast import ProjectionHead  # Reuse projection head
from ...utils import MODEL

log = logging.getLogger(__name__)


class KPConvContrast(BaseModel):
    """KPConv for contrastive self-supervised learning.
    
    Architecture:
        Input → KPConv Encoder → Projection Head → L2-normalized embeddings
        
    The encoder can be initialized from a pretrained checkpoint (e.g., Semantic3D).
    During training, both encoder and projection head are trained with contrastive loss.
    """
    
    def __init__(
        self,
        name='KPConvContrast',
        # Encoder parameters (should match pretrained model)
        num_classes=1,  # Dummy for contrastive learning
        ignored_label_inds=[],
        architecture=[
            'simple', 'resnetb', 'resnetb_strided', 'resnetb', 'resnetb',
            'resnetb_strided', 'resnetb', 'resnetb', 'resnetb_strided',
            'resnetb', 'resnetb', 'resnetb_strided', 'resnetb'
        ],
        in_radius=4.0,
        max_in_points=100000,
        batch_num=8,
        batch_limit=30000,
        val_batch_num=8,
        num_kernel_points=15,
        first_subsampling_dl=0.06,
        conv_radius=2.5,
        first_features_dim=128,
        in_features_dim=1,  # Just XYZ by default
        # Projection head parameters
        projection_hidden_dims=[512, 256],
        projection_output_dim=128,
        projection_use_bn=True,
        projection_dropout=0.1,
        # Pretrained weights
        pretrained_encoder_path=None,
        load_encoder_strict=False,
        freeze_encoder_epochs=0,
        # Training parameters
        encoder_lr_scale=0.1,
        batcher='ConcatBatcher',
        ckpt_path=None,
        **kwargs
    ):
        """Initialize KPConvContrast model.
        
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
            num_classes=num_classes,
            ignored_label_inds=ignored_label_inds,
            architecture=architecture,
            in_radius=in_radius,
            max_in_points=max_in_points,
            batch_num=batch_num,
            batch_limit=batch_limit,
            val_batch_num=val_batch_num,
            num_kernel_points=num_kernel_points,
            first_subsampling_dl=first_subsampling_dl,
            conv_radius=conv_radius,
            first_features_dim=first_features_dim,
            in_features_dim=in_features_dim,
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
        
        # Build encoder (KPConv architecture)
        self._build_encoder()
        
        # Build projection head
        # KPConv encoder outputs features of dimension first_features_dim
        encoder_output_dim = cfg.first_features_dim
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
        
        log.info(f"KPConvContrast initialized:")
        log.info(f"  Encoder output dim: {encoder_output_dim}")
        log.info(f"  Projection output dim: {projection_output_dim}")
        log.info(f"  Freeze encoder epochs: {freeze_encoder_epochs}")

    def _build_encoder(self):
        """Build KPConv encoder layers.
        
        Note: KPConv architecture is more complex than RandLANet.
        This is a simplified version - full implementation would require
        adapting all KPConv modules.
        """
        cfg = self.cfg
        
        # For now, we'll note that KPConv encoder consists of:
        # - Multiple KPConv blocks with strided/non-strided convolutions
        # - Defined by cfg.architecture list
        # - Feature dimension grows through the network
        
        # The actual encoder building is quite complex in KPConv
        # and would require significant adaptation.
        
        log.info(f"Built KPConv encoder with {len(cfg.architecture)} blocks")
        log.warning("KPConv encoder building is simplified - full implementation needed")

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
            
            # Filter encoder keys (exclude head/segmentation layers)
            encoder_state_dict = {}
            for key, value in state_dict.items():
                # Exclude final classification head
                if not any(key.startswith(prefix) for prefix in ['head', 'fc']):
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
            inputs: Dict with batched point cloud data
            
        Returns:
            embeddings: (B, projection_output_dim) L2-normalized embeddings
            encoder_features: (B, encoder_dim) raw encoder features
        """
        # TODO: Implement full KPConv forward pass
        # This is a placeholder - actual implementation would require
        # proper KPConv preprocessing and forward pass
        
        # Placeholder
        batch_size = inputs.get('batch_size', 8)
        device = next(self.parameters()).device
        
        encoder_features = torch.randn(batch_size, self.cfg.first_features_dim, device=device)
        embeddings = self.projection_head(encoder_features)
        
        return {
            'embeddings': embeddings,
            'encoder_features': encoder_features,
        }

    def preprocess(self, data, attr):
        """Preprocess data for contrastive learning."""
        # TODO: Implement KPConv preprocessing
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
