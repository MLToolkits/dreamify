from enum import Enum

from tensorflow.keras.applications import (
    VGG19,
    ConvNeXtXLarge,
    DenseNet121,
    EfficientNetV2L,
    InceptionResNetV2,
    InceptionV3,
    MobileNetV2,
    ResNet152V2,
    Xception,
)


class ModelType(Enum):
    VGG19 = "vgg19"
    CONVNEXT_XL = "convnext_xl"
    DENSENET121 = "densenet121"
    EFFICIENTNET_V2L = "efficientnet_v2l"
    INCEPTION_RESNET_V2 = "inception_resnet_v2"
    INCEPTION_V3 = "inception_v3"
    RESNET152V2 = "resnet152v2"
    XCEPTION = "xception"
    MOBILENET_V2 = "mobilenet_v2"


MODEL_MAP = {
    ModelType.VGG19: VGG19,
    ModelType.CONVNEXT_XL: ConvNeXtXLarge,
    ModelType.DENSENET121: DenseNet121,
    ModelType.EFFICIENTNET_V2L: EfficientNetV2L,
    ModelType.INCEPTION_RESNET_V2: InceptionResNetV2,
    ModelType.INCEPTION_V3: InceptionV3,
    ModelType.RESNET152V2: ResNet152V2,
    ModelType.XCEPTION: Xception,
    ModelType.MOBILENET_V2: MobileNetV2,
}
