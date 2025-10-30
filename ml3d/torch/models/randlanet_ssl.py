import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from .randlanet import RandLANet
from .base_model import BaseModel
from ...utils import MODEL

import logging
log = logging.getLogger(__name__)


class RandLANetSSL(BaseModel):
    """RandLANet wrapper for self-supervised learning tasks.
    
    This model wraps RandLANet's encoder and adds a task-specific head.
    For rotation prediction SSL, it adds global pooling + MLP classifier.
    
    The encoder can be initialized from a pretrained RandLANet checkpoint
    (e.g., trained on Semantic3D for segmentation).
    """

    def __init__(
        self,
        name='RandLANetSSL',
        num_neighbors=16,
        num_layers=4,
        num_points=65536,
        ssl_task='rotation',
        num_rotation_classes=4,
        ignored_label_inds=[],
        sub_sampling_ratio=[4, 4, 4, 4],
        in_channels=3,
        dim_features=8,
        dim_output=[16, 64, 128, 256],
        grid_size=0.06,
        batcher='DefaultBatcher',
        pretrained_encoder_path=None,
        load_encoder_strict=False,
        freeze_encoder_epochs=0,
        ssl_head_dims=[256, 128],
        dropout_rate=0.5,
        augment={},
        **kwargs
    ):
        """Initialize RandLANetSSL.
        
        Args:
            ssl_task: Type of SSL task ('rotation' supported).
            num_rotation_classes: Number of discrete rotation classes (e.g., 4 for 0/90/180/270).
            pretrained_encoder_path: Path to pretrained RandLANet checkpoint.
            load_encoder_strict: If True, require exact match when loading encoder weights.
            freeze_encoder_epochs: Number of initial epochs to freeze encoder (train head only).
            ssl_head_dims: Hidden dimensions for SSL classification head.
            dropout_rate: Dropout rate in SSL head.
        """
        super().__init__(
            name=name,
            num_neighbors=num_neighbors,
            num_layers=num_layers,
            num_points=num_points,
            ignored_label_inds=ignored_label_inds,
            sub_sampling_ratio=sub_sampling_ratio,
            in_channels=in_channels,
            dim_features=dim_features,
            dim_output=dim_output,
            grid_size=grid_size,
            batcher=batcher,
            augment=augment,
            ssl_task=ssl_task,
            num_rotation_classes=num_rotation_classes,
            pretrained_encoder_path=pretrained_encoder_path,
            load_encoder_strict=load_encoder_strict,
            freeze_encoder_epochs=freeze_encoder_epochs,
            ssl_head_dims=ssl_head_dims,
            dropout_rate=dropout_rate,
            **kwargs
        )

        cfg = self.cfg
        self.ssl_task = ssl_task
        self.num_rotation_classes = num_rotation_classes
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.current_epoch = 0

        # Build RandLANet encoder (without segmentation head)
        self._build_encoder()

        # Build SSL task head
        if ssl_task == 'rotation':
            self._build_rotation_head()
        else:
            raise ValueError(f"Unsupported SSL task: {ssl_task}")

        # Load pretrained encoder if provided
        if pretrained_encoder_path:
            self._load_pretrained_encoder(pretrained_encoder_path, load_encoder_strict)

    def _build_encoder(self):
        """Build RandLANet encoder (up to the bottleneck features)."""
        cfg = self.cfg

        # Input projection
        self.fc0 = nn.Linear(cfg.in_channels, cfg.dim_features)
        self.bn0 = nn.BatchNorm2d(cfg.dim_features, eps=1e-6, momentum=0.01)

        # Encoder blocks
        self.encoder = []
        dim_feature = cfg.dim_features
        for i in range(cfg.num_layers):
            # Import LocalFeatureAggregation from randlanet
            from .randlanet import LocalFeatureAggregation
            self.encoder.append(
                LocalFeatureAggregation(dim_feature, cfg.dim_output[i], cfg.num_neighbors)
            )
            dim_feature = 2 * cfg.dim_output[i]

        self.encoder = nn.ModuleList(self.encoder)

        # MLP bottleneck
        from .randlanet import SharedMLP
        self.mlp = SharedMLP(dim_feature, dim_feature, activation_fn=nn.LeakyReLU(0.2))
        
        self.encoder_output_dim = dim_feature

    def _build_rotation_head(self):
        """Build rotation prediction head: global pooling + MLP classifier."""
        cfg = self.cfg
        
        # Global average pooling is applied in forward()
        # Build MLP classifier
        layers = []
        in_dim = self.encoder_output_dim
        
        for hidden_dim in cfg.ssl_head_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(cfg.dropout_rate))
            in_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(in_dim, self.num_rotation_classes))
        
        self.ssl_head = nn.Sequential(*layers)

    def _load_pretrained_encoder(self, checkpoint_path, strict=False):
        """Load pretrained encoder weights from a RandLANet checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file (.pth).
            strict: If True, require exact key match.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            log.warning(f"Pretrained checkpoint not found: {checkpoint_path}")
            return

        log.info(f"Loading pretrained encoder from {checkpoint_path}")
        
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
            
            # Extract state dict (handle different checkpoint formats)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Filter encoder keys (exclude decoder and segmentation head)
            encoder_keys = ['fc0', 'bn0', 'encoder', 'mlp']
            filtered_state = {}
            
            for key, value in state_dict.items():
                # Remove 'module.' prefix if present (DataParallel)
                clean_key = key.replace('module.', '')
                
                # Keep only encoder-related keys
                if any(clean_key.startswith(ek) for ek in encoder_keys):
                    filtered_state[clean_key] = value

            # Load filtered state dict
            missing, unexpected = self.load_state_dict(filtered_state, strict=strict)
            
            if missing:
                log.info(f"Missing keys: {missing}")
            if unexpected:
                log.info(f"Unexpected keys: {unexpected}")
            
            log.info(f"Successfully loaded {len(filtered_state)} encoder parameters")
            
        except Exception as e:
            log.error(f"Failed to load pretrained encoder: {e}")
            raise

    def set_epoch(self, epoch):
        """Update current epoch for conditional freezing logic."""
        self.current_epoch = epoch
        
        # Freeze/unfreeze encoder based on epoch
        if epoch < self.freeze_encoder_epochs:
            self._freeze_encoder()
        else:
            self._unfreeze_encoder()

    def _freeze_encoder(self):
        """Freeze encoder parameters (train SSL head only)."""
        for param in self.fc0.parameters():
            param.requires_grad = False
        for param in self.bn0.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.mlp.parameters():
            param.requires_grad = False
        log.info("Encoder frozen (training SSL head only)")

    def _unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.fc0.parameters():
            param.requires_grad = True
        for param in self.bn0.parameters():
            param.requires_grad = True
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.mlp.parameters():
            param.requires_grad = True
        log.info("Encoder unfrozen (training end-to-end)")

    def forward(self, inputs):
        """Forward pass for SSL task.
        
        Args:
            inputs: dict with keys:
                - 'features': torch.Tensor (B, N, in_channels)
                - 'coords': list of coordinate tensors per layer
                - 'neighbor_indices': list of neighbor indices per layer
                - 'sub_idx': list of subsampling indices per layer
                
        Returns:
            torch.Tensor: (B, num_classes) logits for SSL task
        """
        cfg = self.cfg
        feat = inputs['features'].to(self.device)
        coords_list = [arr.to(self.device) for arr in inputs['coords']]
        neighbor_indices_list = [arr.to(self.device) for arr in inputs['neighbor_indices']]
        subsample_indices_list = [arr.to(self.device) for arr in inputs['sub_idx']]

        # Input projection
        feat = self.fc0(feat).transpose(-2, -1).unsqueeze(-1)  # (B, dim_feature, N, 1)
        feat = self.bn0(feat)
        feat = nn.LeakyReLU(0.2)(feat)

        # Encoder forward
        for i in range(cfg.num_layers):
            feat_encoder_i = self.encoder[i](coords_list[i], feat, neighbor_indices_list[i])
            feat_sampled_i = self.random_sample(feat_encoder_i, subsample_indices_list[i])
            feat = feat_sampled_i

        # Bottleneck MLP
        feat = self.mlp(feat)  # (B, D, N', 1)

        # Global average pooling across points
        feat = feat.squeeze(-1)  # (B, D, N')
        feat_global = torch.mean(feat, dim=2)  # (B, D)

        # SSL head
        logits = self.ssl_head(feat_global)  # (B, num_classes)

        return logits

    @staticmethod
    def random_sample(feature, pool_idx):
        """Random sampling as in RandLANet.
        
        Args:
            feature: [B, d, N, 1] input features
            pool_idx: [B, N', max_num] pooling indices
            
        Returns:
            pool_features: [B, d, N', 1] pooled features
        """
        feature = feature.squeeze(3)
        num_neigh = pool_idx.size()[2]
        batch_size = feature.size()[0]
        d = feature.size()[1]

        pool_idx = torch.reshape(pool_idx, (batch_size, -1))
        pool_idx = pool_idx.unsqueeze(2).expand(batch_size, -1, d)

        feature = feature.transpose(1, 2)
        pool_features = torch.gather(feature, 1, pool_idx)
        pool_features = torch.reshape(pool_features, (batch_size, -1, num_neigh, d))
        pool_features, _ = torch.max(pool_features, 2)
        pool_features = pool_features.transpose(1, 2).unsqueeze(3)

        return pool_features

    def get_optimizer(self, cfg_pipeline):
        """Get optimizer with separate LR for encoder and head."""
        # Separate parameter groups
        encoder_params = (
            list(self.fc0.parameters()) +
            list(self.bn0.parameters()) +
            list(self.encoder.parameters()) +
            list(self.mlp.parameters())
        )
        head_params = list(self.ssl_head.parameters())

        # Use different learning rates if specified
        encoder_lr = cfg_pipeline.get('encoder_lr', cfg_pipeline.learning_rate * 0.1)
        head_lr = cfg_pipeline.learning_rate

        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': head_params, 'lr': head_lr}
        ], weight_decay=cfg_pipeline.get('weight_decay', 1e-4))

        return optimizer

    def preprocess(self, data, attr):
        """Preprocess data (reuse RandLANet's preprocessing logic)."""
        # Import from parent class
        from .randlanet import RandLANet as RandLANetBase
        from sklearn.neighbors import KDTree
        from ...datasets.utils import DataProcessing
        
        cfg = self.cfg
        points = np.array(data['point'][:, 0:3], dtype=np.float32)
        
        # Dummy labels for SSL (not used)
        labels = np.zeros((points.shape[0],), dtype=np.int32)
        
        if 'feat' not in data or data['feat'] is None:
            feat = None
        else:
            feat = np.array(data['feat'], dtype=np.float32)

        split = attr['split']
        processed_data = dict()

        # Grid subsampling
        if feat is None:
            sub_points, sub_labels = DataProcessing.grid_subsampling(
                points, labels=labels, grid_size=cfg.grid_size
            )
            sub_feat = None
        else:
            sub_points, sub_feat, sub_labels = DataProcessing.grid_subsampling(
                points, features=feat, labels=labels, grid_size=cfg.grid_size
            )

        search_tree = KDTree(sub_points)

        processed_data['point'] = sub_points
        processed_data['feat'] = sub_feat
        processed_data['label'] = sub_labels
        processed_data['search_tree'] = search_tree

        if split in ["test", "testing"]:
            proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            processed_data['proj_inds'] = proj_inds

        return processed_data


MODEL._register_module(RandLANetSSL)
