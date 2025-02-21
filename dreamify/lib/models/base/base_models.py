import random

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

from dreamify.lib.models.base.layer_settings import (
    DeepDreamModelLayerSettings,
    ModelType,
    ShallowDreamModelLayerSettings,
)

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


def get_layer_settings(model_name_enum: ModelType, dream_style="deep"):
    if dream_style == "deep":
        return DeepDreamModelLayerSettings[model_name_enum.name].value
    elif dream_style == "shallow":
        return ShallowDreamModelLayerSettings[model_name_enum.name].value
    raise NotImplementedError()


def choose_base_model(model_name: str, dream_style="deep", layer_settings=None):
    if model_name in ["random", "any"]:
        model_name_enum = random.choice(list(ModelType))
    else:
        model_name_enum = ModelType[model_name.upper()]

    model_fn = MODEL_MAP[model_name_enum]
    base_model = model_fn(weights="imagenet", include_top=False)

    if layer_settings is None:
        layer_settings = get_layer_settings(model_name_enum, dream_style)

    return base_model, layer_settings
