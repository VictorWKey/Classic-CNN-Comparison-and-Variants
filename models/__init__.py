from .lenet import CustomLenet
from .lenet_variants import CustomLenet_NoBN, CustomLenet_NoDropout
from .alexnet import AlexNet
from .alexnet_variants import AlexNet_BN, AlexNet_Reduced
from .vgg16 import VGG16
from .vgg16_variants import VGG16_BN, VGG16_GAP

__all__ = [
    'CustomLenet', 'CustomLenet_NoBN', 'CustomLenet_NoDropout',
    'AlexNet', 'AlexNet_BN', 'AlexNet_Reduced', 
    'VGG16', 'VGG16_BN', 'VGG16_GAP'
]