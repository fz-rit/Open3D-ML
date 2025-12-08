"""Networks for torch."""

from .randlanet import RandLANet
from .randlanet_contrast import RandLANetContrast
from .randlanet_da import RandLANetDA
from .kpconv import KPFCNN
from .kpconv_contrast import KPConvContrast
from .kpconv_da import KPFCNNDA
from .point_pillars import PointPillars
from .sparseconvnet import SparseConvUnet
from .point_rcnn import PointRCNN
from .point_transformer import PointTransformer
from .pvcnn import PVCNN

__all__ = [
    'RandLANet', 'RandLANetContrast', 'RandLANetDA', 'KPFCNN', 'KPConvContrast',
    'KPFCNNDA', 'PointPillars', 'PointRCNN', 'SparseConvUnet',
    'PointTransformer', 'PVCNN'
]

try:
    from .openvino_model import OpenVINOModel
    __all__.append("OpenVINOModel")
except Exception:
    pass
