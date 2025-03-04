import random

from dreamify.lib.models.base.constants import MODEL_MAP, ModelType
from dreamify.lib.models.base.layer_settings import ModelLayerNames


def get_layer_settings(model_name_enum: ModelType, dream_style="deep"):
    model_settings = ModelLayerNames[model_name_enum.name]
    if dream_style == "deep":
        return model_settings.deep
    elif dream_style == "shallow":
        return model_settings.deep
    elif dream_style == "classic":
        return model_settings.classic
    else:
        raise NotImplementedError(
            f"Layer settings for {model_name_enum.name} with style {dream_style} not implemented."
        )


def choose_base_model(model_name: str, dream_style="deep", layer_settings=None):
    if model_name in ["random", "any"]:
        model_name_enum = random.choice(list(ModelType))
    else:
        model_name_enum = ModelType[model_name.upper()]

    model_fn = MODEL_MAP[model_name_enum]
    base_model = model_fn(weights="imagenet", include_top=False)

    layer_names = get_layer_settings(model_name_enum, dream_style)

    if layer_settings is not None:
        if layers_settings == "all":
            layer_names = layer_settings
        elif layers_settings == "topmost":
            layer_names = layer_names[len(layer_names) - 2 :]
        elif layers_settings == "bottommost":
            layer_names = layer_names[0:2]
        else:
            layer_names = layer_settings

    return base_model, layer_names


__all__ = [choose_base_model]
