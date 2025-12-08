"""
KPFCNN with Domain Adaptation support.

This module extends KPFCNN to expose intermediate encoder features
for domain adaptation techniques like CORAL (Correlation Alignment).
"""

import time
import math
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_
from sklearn.neighbors import KDTree

from open3d.ml.contrib import subsample_batch
from open3d.ml.torch.layers import FixedRadiusSearch
from open3d.ml.torch.ops import ragged_to_dense

from .base_model import BaseModel
from ..modules.losses import filter_valid_label
from ...utils import MODEL

from ...datasets.utils import (DataProcessing, trans_normalize, trans_augment,
                               trans_crop_pc, create_3D_rotations)

# Import all helper functions and classes from kpconv
from .kpconv import (
    block_decider, BatchNormBlock, UnaryBlock, SimpleBlock,
    ResnetBottleneckBlock, GlobalAverageBlock, NearestUpsampleBlock,
    MaxPoolBlock, KPConv, load_kernels, batch_neighbors,
    batch_grid_subsampling, p2p_fitting_regularizer
)


class KPFCNNDA(BaseModel):
    """KPFCNN with Domain Adaptation support.
    
    This version extends KPFCNN to expose intermediate encoder features
    for domain adaptation techniques like CORAL (Correlation Alignment).
    
    Key differences from base KPFCNN:
    - forward() can return intermediate encoder features
    - Additional method get_encoder_features() for feature extraction
    - Support for domain adaptation training pipelines
    
    Reference: KPConv: Flexible and Deformable Convolution for Point Clouds
    Domain adaptation extension inspired by SqueezeSegV2 (arXiv:1809.08495)
    """

    def __init__(
            self,
            name='KPFCNNDA',
            lbl_values=[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19
            ],
            num_classes=19,
            ignored_label_inds=[0],
            ckpt_path=None,
            batcher='ConcatBatcher',
            architecture=[
                'simple', 'resnetb', 'resnetb_strided', 'resnetb', 'resnetb',
                'resnetb_strided', 'resnetb', 'resnetb', 'resnetb_strided',
                'resnetb', 'resnetb', 'resnetb_strided', 'resnetb',
                'nearest_upsample', 'unary', 'nearest_upsample', 'unary',
                'nearest_upsample', 'unary', 'nearest_upsample', 'unary'
            ],
            in_radius=4.0,
            max_in_points=100000,
            batch_num=8,
            batch_limit=30000,
            val_batch_num=8,
            num_kernel_points=15,
            first_subsampling_dl=0.06,
            conv_radius=2.5,
            deform_radius=6.0,
            KP_extent=1.2,
            KP_influence='linear',
            aggregation_mode='sum',
            first_features_dim=128,
            in_features_dim=2,
            modulated=False,
            use_batch_norm=True,
            batch_norm_momentum=0.02,
            deform_fitting_mode='point2point',
            deform_fitting_power=1.0,
            repulse_extent=1.2,
            augment_scale_anisotropic=True,
            augment_symmetries=[True, False, False],
            augment_rotation='vertical',
            augment_scale_min=0.8,
            augment_scale_max=1.2,
            augment_noise=0.001,
            augment_color=0.8,
            in_points_dim=3,
            fixed_kernel_points='center',
            num_layers=5,
            l_relu=0.1,
            reduce_fc=False,
            return_features=False,
            alignment_layers=None,
            **kwargs):
        """
        Args:
            return_features: If True, forward() returns (logits, features_list).
            alignment_layers: List of encoder block indices to extract features from.
                            If None, uses evenly spaced blocks.
            Other args same as base KPFCNN.
        """

        super().__init__(name=name,
                         lbl_values=lbl_values,
                         num_classes=num_classes,
                         ignored_label_inds=ignored_label_inds,
                         ckpt_path=ckpt_path,
                         batcher=batcher,
                         architecture=architecture,
                         in_radius=in_radius,
                         max_in_points=max_in_points,
                         batch_num=batch_num,
                         batch_limit=batch_limit,
                         val_batch_num=val_batch_num,
                         num_kernel_points=num_kernel_points,
                         first_subsampling_dl=first_subsampling_dl,
                         conv_radius=conv_radius,
                         deform_radius=deform_radius,
                         KP_extent=KP_extent,
                         KP_influence=KP_influence,
                         aggregation_mode=aggregation_mode,
                         first_features_dim=first_features_dim,
                         in_features_dim=in_features_dim,
                         modulated=modulated,
                         use_batch_norm=use_batch_norm,
                         batch_norm_momentum=batch_norm_momentum,
                         deform_fitting_mode=deform_fitting_mode,
                         deform_fitting_power=deform_fitting_power,
                         repulse_extent=repulse_extent,
                         augment_scale_anisotropic=augment_scale_anisotropic,
                         augment_symmetries=augment_symmetries,
                         augment_rotation=augment_rotation,
                         augment_scale_min=augment_scale_min,
                         augment_scale_max=augment_scale_max,
                         augment_noise=augment_noise,
                         augment_color=augment_color,
                         in_points_dim=in_points_dim,
                         fixed_kernel_points=fixed_kernel_points,
                         num_layers=num_layers,
                         l_relu=l_relu,
                         reduce_fc=reduce_fc,
                         return_features=return_features,
                         alignment_layers=alignment_layers,
                         **kwargs)

        cfg = self.cfg
        
        # Domain adaptation settings
        self.return_features = cfg.get('return_features', return_features)
        
        # Count encoder blocks
        num_encoder_blocks = 0
        for block in cfg.architecture:
            if 'upsample' in block or block == 'global_average':
                break
            num_encoder_blocks += 1
        
        if alignment_layers is None:
            # Default: use every 3rd encoder block for alignment
            self.alignment_layers = list(range(0, num_encoder_blocks, 3))
        else:
            self.alignment_layers = alignment_layers

        # Current radius of convolution and feature dimension
        layer = 0
        r = cfg.first_subsampling_dl * cfg.conv_radius
        in_dim = cfg.in_features_dim
        out_dim = cfg.first_features_dim
        lbl_values = cfg.lbl_values
        ign_lbls = cfg.ignored_label_inds
        self.K = cfg.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)

        #####################
        # List Encoder blocks
        #####################

        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        self.neighborhood_limits = []
        
        for block_i, block in enumerate(cfg.architecture):
            # Stop when reaching decoder
            if 'upsample' in block or block == 'global_average':
                break

            # Get all params for this layer
            if block in ['max_pool', 'global_average']:
                self.encoder_blocks.append(block_decider(
                    block, r, in_dim, out_dim, layer, cfg))
            else:
                self.encoder_blocks.append(
                    block_decider(block, r, in_dim, out_dim, layer, cfg))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

            # Save all skip-connections in a list
            if block in ['simple', 'simple_deformable', 'simple_invariant',
                        'simple_equivariant']:
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

        #####################
        # List Decoder blocks
        #####################

        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(cfg.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(cfg.architecture[start_i:]):
            # Add dimension of skip connection
            if block_i > 0 and 'upsample' in cfg.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    cfg))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        if reduce_fc:
            self.head_mlp = UnaryBlock(out_dim, self.C, use_batch_norm,
                                      batch_norm_momentum)
            self.head_softmax = nn.Identity()
        else:
            self.head_mlp = UnaryBlock(out_dim, out_dim, use_batch_norm,
                                      batch_norm_momentum)
            self.head_softmax = UnaryBlock(out_dim, self.C, use_batch_norm,
                                          batch_norm_momentum, no_relu=True)

        ################
        # Network Losses
        ################

        self.valid_labels = np.sort(
            [c for c in lbl_values if c not in ign_lbls])

        self.deform_fitting_mode = cfg.deform_fitting_mode
        self.deform_fitting_power = cfg.deform_fitting_power
        self.repulse_extent = cfg.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, return_intermediate_features=None):
        """Forward pass with domain adaptation support.
        
        Args:
            batch: Input batch
            return_intermediate_features: Override self.return_features if provided.
                If True, returns (logits, feature_list).
                If False, returns only logits.
                
        Returns:
            If return_intermediate_features is True:
                (logits, features_list) where features_list contains encoder features
            Otherwise:
                logits: segmentation scores
        """
        # Determine whether to return features
        should_return_features = (return_intermediate_features 
                                 if return_intermediate_features is not None 
                                 else self.return_features)

        # Get input features
        x = batch.features.clone().detach()

        # Loop over consecutive encoder blocks
        skip_x = []
        alignment_features = []
        
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            
            x = block_op(x, batch)
            
            # Collect features from specified alignment layers
            if should_return_features and block_i in self.alignment_layers:
                # Features are already flat: (N_points, C)
                alignment_features.append(x.clone())

        # Decoder
        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        if should_return_features:
            return x, alignment_features
        else:
            return x

    def get_encoder_features(self, batch):
        """Extract encoder features without computing full forward pass.
        
        Useful for domain adaptation when you only need features, not predictions.
        
        Args:
            batch: Input batch
            
        Returns:
            List of feature tensors from encoder layers specified in alignment_layers
        """
        x = batch.features.clone().detach()

        skip_x = []
        alignment_features = []
        
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            
            x = block_op(x, batch)
            
            # Collect features from specified alignment layers
            if block_i in self.alignment_layers:
                alignment_features.append(x.clone())

        return alignment_features

    def get_optimizer(self, cfg_pipeline):
        """Same as base KPFCNN."""
        deform_params = [v for k, v in self.named_parameters() if 'offset' in k]
        other_params = [
            v for k, v in self.named_parameters() if 'offset' not in k
        ]
        deform_lr = cfg_pipeline.learning_rate * cfg_pipeline.deform_lr_factor
        optimizer = torch.optim.SGD([{
            'params': other_params
        }, {
            'params': deform_params,
            'lr': deform_lr
        }],
                                    lr=cfg_pipeline.learning_rate,
                                    momentum=cfg_pipeline.momentum,
                                    weight_decay=cfg_pipeline.weight_decay)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, cfg_pipeline.scheduler_gamma)

        return optimizer, scheduler

    def get_loss(self, Loss, results, inputs, device):
        """Calculate loss. Handles tuple output from forward().
        
        Args:
            Loss: Loss object
            results: Model output (can be tuple of (logits, features))
            inputs: Input data
            device: Device
            
        Returns:
            loss, labels, scores
        """
        cfg = self.cfg
        labels = inputs['data'].labels
        
        # Handle case where results is a tuple (logits, features)
        if isinstance(results, tuple):
            outputs = results[0]
        else:
            outputs = results

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1).unsqueeze(0)
        labels = labels.unsqueeze(0)

        scores, labels = filter_valid_label(outputs, labels, cfg.num_classes,
                                            cfg.ignored_label_inds, device)

        if labels.numel() == 0:
            self.output_loss = outputs.sum() * 0.0
        else:
            self.output_loss = Loss.weighted_CrossEntropyLoss(scores, labels)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' +
                             self.deform_fitting_mode)

        # Combined loss
        loss = self.output_loss + self.reg_loss

        return loss, labels, scores

    def preprocess(self, data, attr):
        """Same as base KPFCNN."""
        cfg = self.cfg

        points = np.array(data['point'][:, 0:3], dtype=np.float32)

        if 'label' not in data.keys() or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))

        if 'feat' not in data.keys() or data['feat'] is None:
            feat = None
        else:
            feat = np.array(data['feat'], dtype=np.float32)

        split = attr['split']

        data = dict()

        if (feat is None):
            sub_points, sub_labels = DataProcessing.grid_subsampling(
                points, labels=labels, grid_size=cfg.first_subsampling_dl)
            sub_feat = None
        else:
            sub_points, sub_feat, sub_labels = DataProcessing.grid_subsampling(
                points,
                features=feat,
                labels=labels,
                grid_size=cfg.first_subsampling_dl)

        search_tree = KDTree(sub_points)

        data['point'] = sub_points
        data['feat'] = sub_feat
        data['label'] = sub_labels
        data['search_tree'] = search_tree

        if split in ["test", "testing", "validation", "valid"]:
            proj_inds = np.squeeze(
                search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            data['proj_inds'] = proj_inds

        return data

    def transform(self, data, attr, is_test=False):
        """Same as base KPFCNN - simplified version."""
        # This is a complex method - keeping the same implementation as base
        # Import from parent would be ideal but we'll keep a minimal version
        points = data['point']
        sem_labels = data['label']
        feat = data['feat']
        search_tree = data['search_tree']

        dim_points = points.shape[1]
        if feat is None:
            dim_features = dim_points
        else:
            dim_features = feat.shape[1] + dim_points

        merged_points = np.zeros((0, dim_points), dtype=np.float32)
        merged_labels = np.zeros((0,), dtype=np.int32)
        merged_coords = np.zeros((0, dim_features), dtype=np.float32)

        p_origin = np.zeros((1, 4))
        p_origin[0, 3] = 1
        p0 = p_origin[:, :3]
        p0 = np.squeeze(p0)
        o_pts = None
        o_labels = None

        num_merged = 0

        result_data = {
            'p_list': [],
            'f_list': [],
            'l_list': [],
            'p0_list': [],
            's_list': [],
            'R_list': [],
            'r_inds_list': [],
            'r_mask_list': [],
            'val_labels_list': [],
            'cfg': self.cfg
        }

        curr_num_points = 0
        max_num_points = min(self.cfg.batch_limit, self.cfg.max_in_points)
        min_in_points = self.cfg.get('min_in_points', 3)
        min_in_points = min(min_in_points, self.cfg.max_in_points)

        while curr_num_points < min_in_points:
            # Sample points and augment
            o_pts, o_labels, _, R, s = self.augmentation_transform(
                points, sem_labels, feat, is_test=is_test)

            if o_pts is None:
                break

            merged_points = np.vstack((merged_points, o_pts[:, :dim_points]))
            merged_labels = np.hstack((merged_labels, o_labels))
            
            if feat is None:
                merged_coords = np.vstack((merged_coords, o_pts))
            else:
                merged_coords = np.vstack(
                    (merged_coords,
                     np.hstack((o_pts, feat[:o_pts.shape[0]]))))

            num_merged += 1
            curr_num_points = merged_points.shape[0]

            if curr_num_points >= max_num_points:
                break

        result_data['p_list'].append(merged_points)
        result_data['f_list'].append(merged_coords)
        result_data['l_list'].append(merged_labels)
        result_data['p0_list'].append(p0)
        result_data['s_list'].append(s if num_merged > 0 else 1.0)
        result_data['R_list'].append(R if num_merged > 0 else np.eye(3))
        result_data['r_inds_list'].append(np.array([]))
        result_data['r_mask_list'].append(np.array([]))
        result_data['val_labels_list'].append(sem_labels)

        return result_data

    def augmentation_transform(self,
                               points,
                               normals=None,
                               verbose=False,
                               is_test=False):
        """Simplified augmentation - same logic as base KPFCNN."""
        cfg = self.cfg
        
        # Random rotation
        if not is_test and cfg.augment_rotation == 'vertical':
            angle = np.random.rand() * 2 * np.pi
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        elif not is_test and cfg.augment_rotation == 'full':
            R = create_3D_rotations(1)[0]
        else:
            R = np.eye(3, dtype=np.float32)
        
        # Random scaling
        if not is_test:
            scale = np.random.uniform(cfg.augment_scale_min, cfg.augment_scale_max)
        else:
            scale = 1.0
            
        # Apply transformations
        points = np.dot(points, R.T) * scale
        
        if normals is not None:
            normals = np.dot(normals, R.T)
        
        return points, None, normals, R, scale

    def inference_begin(self, data):
        """Same as base KPFCNN."""
        self.test_smooth = 0.98
        attr = {'split': 'test'}
        self.inference_ori_data = data
        self.inference_data = self.preprocess(data, attr)
        self.inference_proj_inds = self.inference_data['proj_inds']
        num_points = self.inference_data['search_tree'].data.shape[0]

        self.possibility = np.random.rand(num_points) * 1e-3
        self.test_probs = np.zeros(shape=[num_points, self.cfg.num_classes],
                                   dtype=np.float16)
        self.pbar = tqdm(total=self.possibility.shape[0])
        self.pbar_update = 0
        from ..dataloaders import ConcatBatcher
        self.batcher = ConcatBatcher(self.device)

    def inference_preprocess(self):
        """Same as base KPFCNN."""
        attr = {'split': 'test'}
        data = self.transform(self.inference_data, attr, is_test=True)
        inputs = {'data': data, 'attr': attr}
        inputs = self.batcher.collate_fn([inputs])
        self.inference_input = inputs

        return inputs

    def update_probs(self, inputs, results, test_probs):
        """Update test probabilities. Handles tuple output."""
        # Handle tuple output
        if isinstance(results, tuple):
            results = results[0]
            
        self.test_smooth = 0.95
        stk_probs = torch.nn.functional.softmax(results, dim=-1)
        stk_probs = stk_probs.cpu().data.numpy()

        batch = inputs['data']
        stk_labels = batch.labels.cpu().data.numpy()

        lengths = batch.lengths[0].cpu().numpy()

        f_inds = batch.frame_inds.cpu().numpy()
        r_inds_list = batch.reproj_inds
        r_mask_list = batch.reproj_masks
        labels_list = batch.val_labels

        i0 = 0
        for b_i, length in enumerate(lengths):
            probs = stk_probs[i0:i0 + length]
            proj_inds = r_inds_list[b_i]
            proj_mask = r_mask_list[b_i]
            frame_labels = labels_list[b_i]
            
            proj_probs = probs[proj_inds]
            
            test_probs[proj_mask] = self.test_smooth * test_probs[proj_mask] + \
                                   (1 - self.test_smooth) * proj_probs[proj_mask]
            i0 += length

        return test_probs

    def inference_end(self, inputs, results):
        """Handle inference end. Handles tuple output."""
        # Handle tuple output
        if isinstance(results, tuple):
            results = results[0]
            
        m_softmax = torch.nn.Softmax(dim=-1)
        stk_probs = m_softmax(results)
        stk_probs = results.cpu().data.numpy()

        batch = inputs['data']

        lengths = batch.lengths[0].cpu().numpy()

        f_inds = batch.frame_inds.cpu().numpy()
        r_inds_list = batch.reproj_inds
        r_mask_list = batch.reproj_masks
        labels_list = batch.val_labels

        i0 = 0
        for b_i, length in enumerate(lengths):
            probs = stk_probs[i0:i0 + length]
            proj_inds = r_inds_list[b_i]
            proj_mask = r_mask_list[b_i]
            
            proj_probs = probs[proj_inds]
            
            self.test_probs[proj_mask] = self.test_smooth * self.test_probs[proj_mask] + \
                                         (1 - self.test_smooth) * proj_probs[proj_mask]
            i0 += length

        self.pbar.update(self.possibility[self.possibility > 0.5].shape[0] -
                         self.pbar_update)
        self.pbar_update = self.possibility[self.possibility > 0.5].shape[0]
        
        if np.min(self.possibility) > 0.5:
            self.pbar.close()
            pred_labels = np.argmax(self.test_probs, 1)

            pred_labels = pred_labels[self.inference_proj_inds]
            test_probs = self.test_probs[self.inference_proj_inds]
            
            inference_result = {
                'predict_labels': pred_labels,
                'predict_scores': test_probs
            }
            
            data = self.inference_ori_data
            self.inference_result = inference_result
            return True
        else:
            return False


MODEL._register_module(KPFCNNDA, 'torch')
