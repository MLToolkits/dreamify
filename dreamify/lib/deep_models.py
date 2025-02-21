# def get_layer_settings(model_name: ModelType, dream_style="deep", layer_settings=None):
#     if layer_settings is None and dream_style == "deep":
#         layer_settings = deepdream_model_layer_settings.get(model_name)
#     elif layer_settings is None and dream_style == "shallow":
#         layer_settings = dream_model_layer_settings.get(model_name)
#     else:
#         raise NotImplementedError()

import tensorflow as tf

#     return layer_settings
from tensorflow.keras.applications import InceptionV3


def choose_deep_model(model_name: str = None, dream_style="deep", layer_settings=None):
    names = ["mixed3", "mixed5"]

    base_model = InceptionV3(weights="imagenet", include_top=False)
    layers = [base_model.get_layer(name).output for name in names]
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
    return dream_model, layers
