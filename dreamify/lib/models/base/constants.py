from enum import Enum

# Preprocessor of the base models
# Models
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
    convnext,
    densenet,
    efficientnet_v2,
    inception_resnet_v2,
    inception_v3,
    mobilenet_v2,
    resnet_v2,
    vgg19,
    xception,
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


def PREPROCESSORS():
    return {
        ModelType.VGG19: vgg19.preprocess_input,
        ModelType.CONVNEXT_XL: convnext.preprocess_input,
        ModelType.DENSENET121: densenet.preprocess_input,
        ModelType.EFFICIENTNET_V2L: efficientnet_v2.preprocess_input,
        ModelType.INCEPTION_RESNET_V2: inception_resnet_v2.preprocess_input,
        ModelType.INCEPTION_V3: inception_v3.preprocess_input,
        ModelType.RESNET152V2: resnet_v2.preprocess_input,
        ModelType.XCEPTION: xception.preprocess_input,
        ModelType.MOBILENET_V2: mobilenet_v2.preprocess_input,
    }
