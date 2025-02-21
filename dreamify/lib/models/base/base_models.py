import random

from dreamify.lib.models.base.constants import MODEL_MAP, ModelType
from dreamify.lib.models.base.layer_settings import (
    DeepDreamModelLayerSettings,
    ShallowDreamModelLayerSettings,
)


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