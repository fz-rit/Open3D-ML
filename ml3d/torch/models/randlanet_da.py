import torch
import torch.nn as nn
import numpy as np
import logging

from tqdm import tqdm
from pathlib import Path
from sklearn.neighbors import KDTree

from .base_model import BaseModel
from .randlanet_da_layers import (AttentivePooling, LocalFeatureAggregation,
                                  LocalSpatialEncoding, SharedMLP)
from ..dataloaders import DefaultBatcher
from ...datasets.augment import SemsegAugmentation
from ..modules.losses import filter_valid_label
from ...datasets.utils import DataProcessing
from ...utils import MODEL

log = logging.getLogger(__name__)


class RandLANetDA(BaseModel):
    """RandLANet with Domain Adaptation support.
    
    This version extends RandLANet to expose intermediate encoder features
    for domain adaptation techniques like CORAL (Correlation Alignment).
    
    Based on the architecture from the paper `RandLA-Net: Efficient Semantic 
    Segmentation of Large-Scale Point Clouds <https://arxiv.org/abs/1911.11236>`__.
    
    Domain adaptation extension inspired by SqueezeSegV2 
    (arXiv:1809.08495) for unsupervised domain adaptation.
    
    Key differences from base RandLANet:
    - forward() can return intermediate encoder features
    - Additional method get_encoder_features() for feature extraction
    - Support for domain adaptation training pipelines
    """

    def __init__(
            self,
            name='RandLANetDA',
            num_neighbors=16,
            num_layers=4,
            num_points=4096 * 11,
            num_classes=19,
            ignored_label_inds=[0],
            sub_sampling_ratio=[4, 4, 4, 4],
            in_channels=3,
            dim_features=8,
            dim_output=[16, 64, 128, 256],
            grid_size=0.06,
            batcher='DefaultBatcher',
            ckpt_path=None,
            augment={},
            return_features=False,
            alignment_layers=None,
            **kwargs):
        """
        Args:
            return_features: If True, forward() returns (logits, features_list).
            alignment_layers: List of layer indices to extract features from 
                            for domain alignment. If None, uses all encoder layers.
            Other args same as base RandLANet.
        """

        super().__init__(name=name,
                         num_neighbors=num_neighbors,
                         num_layers=num_layers,
                         num_points=num_points,
                         num_classes=num_classes,
                         ignored_label_inds=ignored_label_inds,
                         sub_sampling_ratio=sub_sampling_ratio,
                         in_channels=in_channels,
                         dim_features=dim_features,
                         dim_output=dim_output,
                         grid_size=grid_size,
                         batcher=batcher,
                         ckpt_path=ckpt_path,
                         augment=augment,
                         return_features=return_features,
                         alignment_layers=alignment_layers,
                         **kwargs)
        cfg = self.cfg
        self.augmenter = SemsegAugmentation(cfg.augment, seed=self.rng)
        
        # Domain adaptation settings
        self.return_features = cfg.get('return_features', return_features)
        if alignment_layers is None:
            # Default: use all encoder layers
            self.alignment_layers = list(range(cfg.num_layers))
        else:
            self.alignment_layers = alignment_layers

        self.fc0 = nn.Linear(cfg.in_channels, cfg.dim_features)
        self.bn0 = nn.BatchNorm2d(cfg.dim_features, eps=1e-6, momentum=0.01)

        # Encoder
        self.encoder = []
        encoder_dim_list = []
        dim_feature = cfg.dim_features
        for i in range(cfg.num_layers):
            self.encoder.append(
                LocalFeatureAggregation(dim_feature, cfg.dim_output[i],
                                        cfg.num_neighbors))
            dim_feature = 2 * cfg.dim_output[i]
            if i == 0:
                encoder_dim_list.append(dim_feature)
            encoder_dim_list.append(dim_feature)

        self.encoder = nn.ModuleList(self.encoder)

        self.mlp = SharedMLP(dim_feature,
                             dim_feature,
                             activation_fn=nn.LeakyReLU(0.2))

        # Decoder
        self.decoder = []
        for i in range(cfg.num_layers):
            self.decoder.append(
                SharedMLP(encoder_dim_list[-i - 2] + dim_feature,
                          encoder_dim_list[-i - 2],
                          transpose=True,
                          activation_fn=nn.LeakyReLU(0.2)))
            dim_feature = encoder_dim_list[-i - 2]

        self.decoder = nn.ModuleList(self.decoder)

        self.fc1 = nn.Sequential(
            SharedMLP(dim_feature, 64, activation_fn=nn.LeakyReLU(0.2)),
            SharedMLP(64, 32, activation_fn=nn.LeakyReLU(0.2)), nn.Dropout(0.5),
            SharedMLP(32, cfg.num_classes, bn=False))

    def preprocess(self, data, attr):
        """Same as base RandLANet."""
        cfg = self.cfg

        points = np.array(data['point'][:, 0:3], dtype=np.float32)

        if 'label' not in data or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))

        if 'feat' not in data or data['feat'] is None:
            feat = None
        else:
            feat = np.array(data['feat'], dtype=np.float32)
        
        if 'intensity' in data and data['intensity'] is not None:
            intensity = np.array(data['intensity'], dtype=np.float32).reshape(-1, 1)
        else:
            intensity = None

        split = attr['split']
        data = dict()

        if feat is None and intensity is None:
            sub_points, sub_labels = DataProcessing.grid_subsampling(
                points, labels=labels, grid_size=cfg.grid_size)
            sub_feat = None
            sub_intensity = None
        elif feat is not None:
            sub_points, sub_feat, sub_labels = DataProcessing.grid_subsampling(
                points, features=feat, labels=labels, grid_size=cfg.grid_size)
            sub_intensity = None
        elif intensity is not None:
            sub_points, sub_intensity, sub_labels = DataProcessing.grid_subsampling(
                points, features=intensity, labels=labels, grid_size=cfg.grid_size)
            sub_feat = None

        search_tree = KDTree(sub_points)

        data['point'] = sub_points
        data['feat'] = sub_feat
        data['intensity'] = sub_intensity
        data['label'] = sub_labels
        data['search_tree'] = search_tree

        if split in ["test", "testing"]:
            proj_inds = np.squeeze(
                search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            data['proj_inds'] = proj_inds

        return data

    def transform(self, data, attr, min_possibility_idx=None):
        """Same as base RandLANet."""
        if torch.utils.data.get_worker_info():
            seedseq = np.random.SeedSequence(
                torch.utils.data.get_worker_info().seed +
                torch.utils.data.get_worker_info().id)
            rng = np.random.default_rng(seedseq.spawn(1)[0])
        else:
            rng = self.rng

        cfg = self.cfg
        inputs = dict()

        pc = data['point'].copy()
        label = data['label'].copy()
        
        if cfg.in_channels == 4:
            if 'intensity' in data and data['intensity'] is not None:
                feat = data['intensity'].copy().reshape(-1, 1)
            elif data.get('feat') is not None:
                feat = data['feat'].copy()
            else:
                feat = None
        elif cfg.in_channels == 6:
            feat = data['feat'].copy() if data['feat'] is not None else None
        else:
            feat = None
        
        tree = data['search_tree']

        pc, selected_idxs, center_point = self.trans_point_sampler(
            pc=pc,
            feat=feat,
            label=label,
            search_tree=tree,
            num_points=self.cfg.num_points)

        label = label[selected_idxs]

        if feat is not None:
            feat = feat[selected_idxs]

        augment_cfg = self.cfg.get('augment', {}).copy()
        val_augment_cfg = {}
        if 'recenter' in augment_cfg:
            val_augment_cfg['recenter'] = augment_cfg.pop('recenter')
        if 'normalize' in augment_cfg:
            val_augment_cfg['normalize'] = augment_cfg.pop('normalize')

        self.augmenter.augment(pc, feat, label, val_augment_cfg, seed=rng)

        if attr['split'] in ['training', 'train']:
            pc, feat, label = self.augmenter.augment(pc,
                                                     feat,
                                                     label,
                                                     augment_cfg,
                                                     seed=rng)

        if feat is None:
            feat = pc.copy()
        else:
            feat = np.concatenate([pc, feat], axis=1)

        if cfg.in_channels != feat.shape[1]:
            raise RuntimeError(
                "Wrong feature dimension, please update in_channels(3 + feature_dimension) in config"
            )

        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):
            neighbour_idx = DataProcessing.knn_search(pc, pc, cfg.num_neighbors)

            sub_points = pc[:pc.shape[0] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:pc.shape[0] // cfg.sub_sampling_ratio[i], :]
            up_i = DataProcessing.knn_search(sub_points, pc, 1)
            input_points.append(pc)
            input_neighbors.append(neighbour_idx.astype(np.int64))
            input_pools.append(pool_i.astype(np.int64))
            input_up_samples.append(up_i.astype(np.int64))
            pc = sub_points

        inputs['coords'] = input_points
        inputs['neighbor_indices'] = input_neighbors
        inputs['sub_idx'] = input_pools
        inputs['interp_idx'] = input_up_samples
        inputs['features'] = feat
        inputs['point_inds'] = selected_idxs
        inputs['labels'] = label.astype(np.int64)

        return inputs

    def forward(self, inputs, return_intermediate_features=None):
        """Forward pass for RandLANet with domain adaptation support.

        Args:
            inputs: Input dictionary with point cloud data
            return_intermediate_features: Override self.return_features if provided.
                If True, returns (logits, feature_list).
                If False, returns only logits.

        Returns:
            If return_intermediate_features is True:
                (scores, features_list) where features_list contains encoder features
            Otherwise:
                scores: (B, N, num_classes) segmentation logits
        """
        cfg = self.cfg
        
        # Determine whether to return features
        should_return_features = (return_intermediate_features 
                                 if return_intermediate_features is not None 
                                 else self.return_features)
        
        feat = inputs['features'].to(self.device)
        coords_list = [arr.to(self.device) for arr in inputs['coords']]
        neighbor_indices_list = [
            arr.to(self.device) for arr in inputs['neighbor_indices']
        ]
        subsample_indices_list = [
            arr.to(self.device) for arr in inputs['sub_idx']
        ]
        interpolation_indices_list = [
            arr.to(self.device) for arr in inputs['interp_idx']
        ]

        feat = self.fc0(feat).transpose(-2, -1).unsqueeze(-1)
        feat = self.bn0(feat)

        l_relu = nn.LeakyReLU(0.2)
        feat = l_relu(feat)

        # Encoder - collect features for domain adaptation
        encoder_feat_list = []
        alignment_features = []  # Features to use for domain alignment
        
        for i in range(cfg.num_layers):
            feat_encoder_i = self.encoder[i](coords_list[i], feat,
                                             neighbor_indices_list[i])
            feat_sampled_i = self.random_sample(feat_encoder_i,
                                                subsample_indices_list[i])
            if i == 0:
                encoder_feat_list.append(feat_encoder_i.clone())
            encoder_feat_list.append(feat_sampled_i.clone())
            
            # Collect features from specified alignment layers
            if should_return_features and i in self.alignment_layers:
                # Flatten spatial dimensions for CORAL: (B, C, N, 1) -> (B*N, C)
                feat_flat = feat_encoder_i.squeeze(3).transpose(1, 2).contiguous()
                feat_flat = feat_flat.view(-1, feat_flat.size(-1))
                alignment_features.append(feat_flat)
            
            feat = feat_sampled_i

        feat = self.mlp(feat)

        # Decoder
        for i in range(cfg.num_layers):
            feat_interpolation_i = self.nearest_interpolation(
                feat, interpolation_indices_list[-i - 1])
            feat_decoder_i = torch.cat(
                [encoder_feat_list[-i - 2], feat_interpolation_i], dim=1)
            feat_decoder_i = self.decoder[i](feat_decoder_i)
            feat = feat_decoder_i

        scores = self.fc1(feat)
        scores = scores.squeeze(3).transpose(1, 2)
        
        if should_return_features:
            return scores, alignment_features
        else:
            return scores

    def get_encoder_features(self, inputs):
        """Extract encoder features without computing full forward pass.
        
        Useful for domain adaptation when you only need features, not predictions.
        
        Args:
            inputs: Input dictionary
            
        Returns:
            List of feature tensors from encoder layers specified in alignment_layers
        """
        cfg = self.cfg
        
        feat = inputs['features'].to(self.device)
        coords_list = [arr.to(self.device) for arr in inputs['coords']]
        neighbor_indices_list = [
            arr.to(self.device) for arr in inputs['neighbor_indices']
        ]
        subsample_indices_list = [
            arr.to(self.device) for arr in inputs['sub_idx']
        ]

        feat = self.fc0(feat).transpose(-2, -1).unsqueeze(-1)
        feat = self.bn0(feat)
        feat = nn.LeakyReLU(0.2)(feat)

        alignment_features = []
        
        for i in range(cfg.num_layers):
            feat_encoder_i = self.encoder[i](coords_list[i], feat,
                                             neighbor_indices_list[i])
            
            # Collect features from specified alignment layers
            if i in self.alignment_layers:
                feat_flat = feat_encoder_i.squeeze(3).transpose(1, 2).contiguous()
                feat_flat = feat_flat.view(-1, feat_flat.size(-1))
                alignment_features.append(feat_flat)
            
            feat_sampled_i = self.random_sample(feat_encoder_i,
                                                subsample_indices_list[i])
            feat = feat_sampled_i

        return alignment_features

    @staticmethod
    def random_sample(feature, pool_idx):
        """Same as base RandLANet."""
        feature = feature.squeeze(3)
        num_neigh = pool_idx.size()[2]
        batch_size = feature.size()[0]
        d = feature.size()[1]

        pool_idx = torch.reshape(pool_idx, (batch_size, -1))
        pool_idx = pool_idx.unsqueeze(2).expand(batch_size, -1, d)

        feature = feature.transpose(1, 2)
        pool_features = torch.gather(feature, 1, pool_idx)
        pool_features = torch.reshape(pool_features,
                                      (batch_size, -1, num_neigh, d))
        pool_features, _ = torch.max(pool_features, 2, keepdim=True)
        pool_features = pool_features.permute(0, 3, 1, 2)

        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """Same as base RandLANet."""
        feature = feature.squeeze(3)
        d = feature.size(1)
        batch_size = interp_idx.size()[0]
        up_num_points = interp_idx.size()[1]

        interp_idx = torch.reshape(interp_idx, (batch_size, up_num_points))
        interp_idx = interp_idx.unsqueeze(1).expand(batch_size, d, -1)

        interpolatedim_features = torch.gather(feature, 2, interp_idx)
        interpolatedim_features = interpolatedim_features.unsqueeze(3)
        return interpolatedim_features

    def get_optimizer(self, cfg_pipeline):
        """Same as base RandLANet."""
        optimizer = torch.optim.Adam(self.parameters(),
                                     **cfg_pipeline.optimizer)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, cfg_pipeline.scheduler_gamma)
        return optimizer, scheduler

    def check_finite(self, reset_bn=False, raise_on_nan=True, prefix=""):
        """Check parameters and BN running stats for NaN/Inf.

        Args:
            reset_bn: If True, reset any non-finite BN running stats to defaults.
            raise_on_nan: If True, raise RuntimeError when non-finite values are found.
            prefix: Text prefix for logging/exception context.
        """

        issues = []

        for name, param in self.named_parameters():
            if param is not None and not torch.isfinite(param).all():
                issues.append(f"param:{name}")

        for name, module in self.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                rm = getattr(module, 'running_mean', None)
                rv = getattr(module, 'running_var', None)
                bad_rm = rm is not None and not torch.isfinite(rm).all()
                bad_rv = rv is not None and not torch.isfinite(rv).all()
                if bad_rm or bad_rv:
                    issues.append(f"bn:{name}")
                    if reset_bn:
                        if bad_rm:
                            module.running_mean.data.zero_()
                        if bad_rv:
                            module.running_var.data.fill_(1.0)
                        log.warning(f"{prefix} reset BN stats for {name} (bad running stats)")

        if issues:
            msg = f"{prefix} non-finite values found in: {issues}"
            if raise_on_nan:
                raise RuntimeError(msg)
            log.error(msg)
        return not issues

    def get_loss(self, Loss, results, inputs, device):
        """Calculate the loss on output of the model.
        
        Same as base RandLANet but compatible with (logits, features) tuple output.
        """
        cfg = self.cfg
        labels = inputs['data']['labels']
        
        # Handle case where results is a tuple (logits, features)
        if isinstance(results, tuple):
            results = results[0]  # Extract logits

        scores, labels = filter_valid_label(results, labels, cfg.num_classes,
                                            cfg.ignored_label_inds, device)

        # Fail fast if logits already contain NaN/Inf before computing loss
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            def _stats(t):
                finite = torch.isfinite(t)
                numel = t.numel()
                num_nan = torch.isnan(t).sum().item()
                num_inf = torch.isinf(t).sum().item()
                if finite.any():
                    t_f = t[finite]
                    return {
                        'shape': tuple(t.shape),
                        'dtype': str(t.dtype),
                        'min': t_f.min().item(),
                        'max': t_f.max().item(),
                        'mean': t_f.float().mean().item(),
                        'num_nan': num_nan,
                        'num_inf': num_inf,
                        'numel': numel,
                    }
                return {
                    'shape': tuple(t.shape),
                    'dtype': str(t.dtype),
                    'min': None,
                    'max': None,
                    'mean': None,
                    'num_nan': num_nan,
                    'num_inf': num_inf,
                    'numel': numel,
                }

            unique_labels = torch.unique(labels).tolist()
            raw_labels = inputs['data'].get('labels', None)
            raw_label_stats = _stats(raw_labels) if raw_labels is not None else None
            scores_stats = _stats(scores)
            features_stats = _stats(inputs['data']['features']) if 'features' in inputs['data'] else None
            coords_stats = _stats(inputs['data']['coords'][0]) if 'coords' in inputs['data'] and len(inputs['data']['coords']) > 0 else None

            log.error("NaN/Inf detected in logits before loss computation. Raising to stop training.")
            log.error(f"  filtered labels unique: {unique_labels}")
            log.error(f"  raw label stats: {raw_label_stats}")
            log.error(f"  scores stats: {scores_stats}")
            log.error(f"  features stats: {features_stats}")
            log.error(f"  coords[0] stats: {coords_stats}")

            raise RuntimeError(
                "NaN/Inf detected in logits before loss. "
                f"Filtered labels: {unique_labels}, Scores stats: {scores_stats}, "
                f"Raw labels: {raw_label_stats}, Features: {features_stats}, "
                f"Coords[0]: {coords_stats}"
            )

        if labels.numel() == 0:
            # No valid labels - return zero loss
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, labels, scores

        loss = Loss.weighted_CrossEntropyLoss(scores, labels)
        
        # # Protect against NaN/Inf from ignored labels or numerical issues
        # if torch.isnan(loss) or torch.isinf(loss):
        #     def _stats(t):
        #         finite = torch.isfinite(t)
        #         numel = t.numel()
        #         num_nan = torch.isnan(t).sum().item()
        #         num_inf = torch.isinf(t).sum().item()
        #         if finite.any():
        #             t_f = t[finite]
        #             return {
        #                 'shape': tuple(t.shape),
        #                 'dtype': str(t.dtype),
        #                 'min': t_f.min().item(),
        #                 'max': t_f.max().item(),
        #                 'mean': t_f.float().mean().item(),
        #                 'num_nan': num_nan,
        #                 'num_inf': num_inf,
        #                 'numel': numel,
        #             }
        #         return {
        #             'shape': tuple(t.shape),
        #             'dtype': str(t.dtype),
        #             'min': None,
        #             'max': None,
        #             'mean': None,
        #             'num_nan': num_nan,
        #             'num_inf': num_inf,
        #             'numel': numel,
        #         }

            # unique_labels = torch.unique(labels).tolist()
            # raw_labels = inputs['data'].get('labels', None)
            # raw_label_stats = _stats(raw_labels) if raw_labels is not None else None

            # scores_stats = _stats(scores)
            # loss_value = loss.item() if torch.isfinite(loss) else str(loss)

            # features_stats = None
            # if 'features' in inputs['data']:
            #     features_stats = _stats(inputs['data']['features'])

            # coords_stats = None
            # if 'coords' in inputs['data'] and len(inputs['data']['coords']) > 0:
            #     coords_stats = _stats(inputs['data']['coords'][0])

            # log.error("NaN/Inf detected in segmentation loss. Raising to stop training.")
            # log.error(f"  loss: {loss_value}")
            # log.error(f"  filtered labels unique: {unique_labels}")
            # log.error(f"  raw label stats: {raw_label_stats}")
            # log.error(f"  scores stats: {scores_stats}")
            # log.error(f"  features stats: {features_stats}")
            # log.error(f"  coords[0] stats: {coords_stats}")

            # raise RuntimeError(
            #     "NaN/Inf detected in segmentation loss. "
            #     f"Filtered labels: {unique_labels}, Scores stats: {scores_stats}, "
            #     f"Raw labels: {raw_label_stats}, Features: {features_stats}, "
            #     f"Coords[0]: {coords_stats}"
            # )

        return loss, labels, scores

    def inference_begin(self, data):
        """Same as base RandLANet."""
        self.test_smooth = 0.95
        attr = {'split': 'test'}
        self.inference_ori_data = data
        self.inference_data = self.preprocess(data, attr)
        self.inference_proj_inds = self.inference_data['proj_inds']
        num_points = self.inference_data['search_tree'].data.shape[0]
        self.possibility = self.rng.random(num_points) * 1e-3
        self.test_probs = np.zeros(shape=[num_points, self.cfg.num_classes],
                                   dtype=np.float16)
        self.pbar = tqdm(total=self.possibility.shape[0])
        self.pbar_update = 0
        self.batcher = DefaultBatcher()

    def inference_preprocess(self):
        """Same as base RandLANet."""
        min_possibility_idx = np.argmin(self.possibility)
        attr = {'split': 'test'}
        data = self.transform(self.inference_data, attr, min_possibility_idx)
        inputs = {'data': data, 'attr': attr}
        inputs = self.batcher.collate_fn([inputs])
        self.inference_input = inputs
        return inputs

    def inference_end(self, inputs, results):
        """Same as base RandLANet but handles tuple output."""
        # Handle case where results is a tuple (logits, features)
        if isinstance(results, tuple):
            results = results[0]
        
        results = torch.reshape(results, (-1, self.cfg.num_classes))
        m_softmax = torch.nn.Softmax(dim=-1)
        results = m_softmax(results)
        results = results.cpu().data.numpy()
        probs = np.reshape(results, [-1, self.cfg.num_classes])

        pred_l = np.argmax(probs, 1)

        inds = inputs['data']['point_inds']
        self.test_probs[inds] = self.test_smooth * self.test_probs[inds] + (
            1 - self.test_smooth) * probs

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
            acc = (pred_labels == data['label'] - 1).mean()

            self.inference_result = inference_result
            return True
        else:
            return False

    def update_probs(self, inputs, results, test_probs):
        """Update test probabilities. Same as base RandLANet."""
        # Handle tuple output
        if isinstance(results, tuple):
            results = results[0]
            
        self.test_smooth = 0.95

        for b in range(results.size()[0]):
            result = torch.reshape(results[b], (-1, self.cfg.num_classes))
            probs = torch.nn.functional.softmax(result, dim=-1)
            probs = probs.cpu().data.numpy()
            inds = inputs['data']['point_inds'][b]

            test_probs[inds] = self.test_smooth * test_probs[inds] + (
                1 - self.test_smooth) * probs

        return test_probs


MODEL._register_module(RandLANetDA, 'torch')
