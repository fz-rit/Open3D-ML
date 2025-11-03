"""
HarvardForest3D dataset for contrastive learning (PointContrast-style).

This dataset generates pairs of augmented views from the same point cloud
and tracks correspondences between them for contrastive learning.
"""

import numpy as np
from pathlib import Path
import logging

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

from .harvardforest3d import HarvardForest3D, HarvardForest3DSplit
from .base_dataset import BaseDatasetSplit
from ..utils import DATASET

log = logging.getLogger(__name__)


class HarvardForest3DContrastive(HarvardForest3D):
    """HarvardForest3D dataset for contrastive self-supervised learning.
    
    Generates two augmented views of each point cloud sample and tracks
    correspondences for PointContrast-style training.
    
    Key features:
    - Two augmented views per sample (view1, view2)
    - Correspondence tracking between views
    - Multiple augmentation strategies (rotation, scale, jitter, dropout)
    - Support for both RandLANet and KPConv encoders
    """

    def __init__(
        self,
        dataset_path,
        name='HarvardForest3DContrastive',
        cache_dir='./logs/cache_harvardforest_contrast',
        use_cache=False,
        num_points=8192,  # Points to sample per cloud
        val_split_ratio=0.1,
        val_split_seed=42,
        test_split_ratio=0.0,
        test_result_folder='./test',
        use_intensity=False,
        use_rgb=False,
        # Augmentation parameters
        augment_rotation_range=360,  # Full rotation around Z-axis
        augment_scale_min=0.8,
        augment_scale_max=1.2,
        augment_jitter_std=0.01,
        augment_dropout_ratio=0.2,  # Drop 20% of points randomly
        augment_translate_range=0.5,  # Translate by ±0.5m
        # Correspondence parameters
        correspondence_threshold=0.05,  # 5cm threshold for matching points
        min_correspondences=512,  # Minimum correspondences per pair
        **kwargs,
    ):
        """Initialize contrastive learning dataset.
        
        Args:
            dataset_path: Path to HarvardForest3D .las files
            num_points: Number of points to sample from each cloud
            augment_rotation_range: Rotation range in degrees (0-360)
            augment_scale_min/max: Random scaling range
            augment_jitter_std: Gaussian noise std for jittering
            augment_dropout_ratio: Fraction of points to randomly drop
            augment_translate_range: Random translation range in meters
            correspondence_threshold: Distance threshold for matching points
            min_correspondences: Minimum number of correspondences required
        """
        if not HAS_LASPY:
            raise ImportError("laspy is required. Install: pip install laspy")
        
        super().__init__(
            dataset_path=dataset_path,
            name=name,
            cache_dir=cache_dir,
            use_cache=use_cache,
            num_points=num_points,
            val_split_ratio=val_split_ratio,
            val_split_seed=val_split_seed,
            test_split_ratio=test_split_ratio,
            test_result_folder=test_result_folder,
            use_intensity=use_intensity,
            use_rgb=use_rgb,
            **kwargs,
        )
        
        # Store num_points for easy access
        self.num_points = num_points
        
        # Augmentation parameters
        self.augment_rotation_range = augment_rotation_range
        self.augment_scale_min = augment_scale_min
        self.augment_scale_max = augment_scale_max
        self.augment_jitter_std = augment_jitter_std
        self.augment_dropout_ratio = augment_dropout_ratio
        self.augment_translate_range = augment_translate_range
        
        # Correspondence parameters
        self.correspondence_threshold = correspondence_threshold
        self.min_correspondences = min_correspondences
        
        log.info(f"HarvardForest3DContrastive initialized for contrastive learning")
        log.info(f"  Augmentation: rotation=±{augment_rotation_range}°, "
                f"scale=[{augment_scale_min},{augment_scale_max}], "
                f"jitter={augment_jitter_std}, dropout={augment_dropout_ratio}")

    def get_split(self, split):
        """Get dataset split for contrastive learning."""
        return HarvardForest3DContrastiveSplit(self, split=split)


class HarvardForest3DContrastiveSplit(HarvardForest3DSplit):
    """Split for contrastive learning with augmentation and correspondence tracking."""

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)
        
        # Copy augmentation parameters from parent dataset
        self.augment_rotation_range = dataset.augment_rotation_range
        self.augment_scale_min = dataset.augment_scale_min
        self.augment_scale_max = dataset.augment_scale_max
        self.augment_jitter_std = dataset.augment_jitter_std
        self.augment_dropout_ratio = dataset.augment_dropout_ratio
        self.augment_translate_range = dataset.augment_translate_range
        self.correspondence_threshold = dataset.correspondence_threshold
        self.min_correspondences = dataset.min_correspondences
        
        log.info(f"Contrastive split '{split}' ready with {len(self.path_list)} samples")

    def __getitem__(self, idx):
        """Get a contrastive pair: two augmented views of the same cloud.
        
        Returns:
            dict with keys:
                - 'point_view1': (N, 3) coordinates for view 1
                - 'point_view2': (N, 3) coordinates for view 2
                - 'name': filename
        """
        # Load original point cloud
        las_file = self.path_list[idx]
        points, features = self._load_las_file(las_file)
        
        # Sample points from original cloud
        if len(points) > self.dataset.num_points:
            indices = np.random.choice(len(points), self.dataset.num_points, replace=False)
        else:
            # Upsample if too few points
            indices = np.random.choice(len(points), self.dataset.num_points, replace=True)
        
        points_base = points[indices]
        
        # Generate two independently augmented views
        # No need to track correspondences for cloud-level contrastive learning
        view1_points = self._augment_view_simple(points_base)
        view2_points = self._augment_view_simple(points_base)
        
        result = {
            'point_view1': view1_points,
            'point_view2': view2_points,
            'name': Path(las_file).stem,
        }
        
        return result

    def _load_las_file(self, las_path):
        """Load point cloud from LAS file.
        
        Returns:
            points: (N, 3) XYZ coordinates
            features: (N, D) features (intensity/RGB) or None
        """
        try:
            las = laspy.read(las_path)
            
            # Get XYZ coordinates
            x = np.array(las.x, dtype=np.float32)
            y = np.array(las.y, dtype=np.float32)
            z = np.array(las.z, dtype=np.float32)
            points = np.stack([x, y, z], axis=1)
            
            # Get features if requested
            features = []
            if self.dataset.use_intensity and hasattr(las, 'intensity'):
                intensity = np.array(las.intensity, dtype=np.float32)
                # Normalize to [0, 1]
                if intensity.max() > intensity.min():
                    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
                features.append(intensity.reshape(-1, 1))
            
            if self.dataset.use_rgb:
                if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                    r = np.array(las.red, dtype=np.float32) / 65535.0
                    g = np.array(las.green, dtype=np.float32) / 65535.0
                    b = np.array(las.blue, dtype=np.float32) / 65535.0
                    features.extend([r.reshape(-1, 1), g.reshape(-1, 1), b.reshape(-1, 1)])
            
            if len(features) > 0:
                features = np.concatenate(features, axis=1)
            else:
                features = None
            
            return points, features
            
        except Exception as e:
            log.error(f"Error loading {las_path}: {e}")
            raise
    
    def _augment_view_simple(self, points):
        """Apply random augmentations to create a view (simplified, no dropout).
        
        Args:
            points: (N, 3) point coordinates
            
        Returns:
            augmented_points: (N, 3) augmented coordinates
        """
        points = points.copy()
        
        # Center points
        centroid = points.mean(axis=0)
        points_centered = points - centroid
        
        # Random rotation around Z-axis
        if self.augment_rotation_range > 0:
            angle = np.random.uniform(0, self.augment_rotation_range) * np.pi / 180.0
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            points_centered = points_centered @ rotation_matrix.T
        
        # Random scaling
        if self.augment_scale_min < self.augment_scale_max:
            scale = np.random.uniform(self.augment_scale_min, self.augment_scale_max)
            points_centered = points_centered * scale
        
        # Random translation
        if self.augment_translate_range > 0:
            translation = np.random.uniform(
                -self.augment_translate_range,
                self.augment_translate_range,
                size=3
            ).astype(np.float32)
            points_centered = points_centered + translation
        
        # Add Gaussian jitter
        if self.augment_jitter_std > 0:
            jitter = np.random.normal(0, self.augment_jitter_std, points_centered.shape)
            points_centered = points_centered + jitter.astype(np.float32)
        
        # Restore to original coordinate frame
        points_augmented = points_centered + centroid
        
        return points_augmented.astype(np.float32)

    def _augment_view(self, points, features, return_indices=True):
        """Apply augmentations to create one view (legacy with correspondence tracking).
        
        Args:
            points: (N, 3) original points
            features: (N, D) features or None
            return_indices: If True, track which original points survived
            
        Returns:
            dict with 'points', 'features', 'original_indices'
        """
        # Track original indices
        original_indices = np.arange(len(points))
        
        # 1. Random point dropout
        if self.augment_dropout_ratio > 0:
            keep_ratio = 1.0 - self.augment_dropout_ratio
            n_keep = max(self.min_correspondences, int(len(points) * keep_ratio))
            keep_indices = np.random.choice(len(points), n_keep, replace=False)
            keep_indices = np.sort(keep_indices)  # Keep order for stability
            points = points[keep_indices]
            if features is not None:
                features = features[keep_indices]
            original_indices = original_indices[keep_indices]
        
        # 2. Center points (important for rotation/scale)
        centroid = points.mean(axis=0)
        points_centered = points - centroid
        
        # 3. Random rotation around Z-axis
        if self.augment_rotation_range > 0:
            angle = np.random.uniform(0, self.augment_rotation_range) * np.pi / 180.0
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            points_centered = points_centered @ rotation_matrix.T
        
        # 4. Random scaling
        if self.augment_scale_min < self.augment_scale_max:
            scale = np.random.uniform(self.augment_scale_min, self.augment_scale_max)
            points_centered = points_centered * scale
        
        # 5. Random translation
        if self.augment_translate_range > 0:
            translation = np.random.uniform(
                -self.augment_translate_range,
                self.augment_translate_range,
                size=3
            ).astype(np.float32)
            points_centered = points_centered + translation
        
        # 6. Add Gaussian jitter
        if self.augment_jitter_std > 0:
            jitter = np.random.normal(0, self.augment_jitter_std, points_centered.shape)
            points_centered = points_centered + jitter.astype(np.float32)
        
        # Restore to original coordinate frame (add centroid back)
        points_augmented = points_centered + centroid
        
        return {
            'points': points_augmented.astype(np.float32),
            'features': features,
            'original_indices': original_indices,
        }

    def _find_correspondences(self, indices1, indices2):
        """Find corresponding points between two views based on original indices.
        
        Args:
            indices1: (N1,) original indices for view 1
            indices2: (N2,) original indices for view 2
            
        Returns:
            correspondences: (K, 2) array where correspondences[i] = [idx1, idx2]
                            means point idx1 in view1 corresponds to point idx2 in view2
        """
        # Find intersection of indices (points that survived dropout in both views)
        common_indices = np.intersect1d(indices1, indices2)
        
        if len(common_indices) < self.min_correspondences:
            log.warning(
                f"Only {len(common_indices)} correspondences found, "
                f"less than minimum {self.min_correspondences}. "
                "Consider reducing augment_dropout_ratio."
            )
        
        # Create correspondence pairs
        correspondences = []
        for orig_idx in common_indices:
            idx1 = np.where(indices1 == orig_idx)[0][0]
            idx2 = np.where(indices2 == orig_idx)[0][0]
            correspondences.append([idx1, idx2])
        
        correspondences = np.array(correspondences, dtype=np.int64)
        
        return correspondences


# Register the dataset
DATASET._register_module(HarvardForest3DContrastive)
