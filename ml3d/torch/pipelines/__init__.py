"""3D ML pipelines for torch."""

from .semantic_segmentation import SemanticSegmentation
from .object_detection import ObjectDetection
from .ssl_rotation import SSLRotation

__all__ = ['SemanticSegmentation', 'ObjectDetection', 'SSLRotation']
