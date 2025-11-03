"""Networks for torch."""

from .randlanet import RandLANet
from .randlanet_contrast import RandLANetContrast
from .kpconv import KPFCNN
from .kpconv_contrast import KPConvContrast
from .point_pillars import PointPillars
from .sparseconvnet import SparseConvUnet
from .point_rcnn import PointRCNN
from .point_transformer import PointTransformer
from .pvcnn import PVCNN

__all__ = [
    'RandLANet', 'RandLANetContrast', 'KPFCNN', 'KPConvContrast',
    'PointPillars', 'PointRCNN', 'SparseConvUnet',
    'PointTransformer', 'PVCNN'
]

try:
    from .openvino_model import OpenVINOModel
    __all__.append("OpenVINOModel")
except Exception:
    pass
