import tensorflow as tf
from tensorflow.keras import Model

from dreamify.lib.models.base import choose_base_model


class FeatureExtractor:
    def __init__(self, model_name, dream_style, layer_settings):
        self.model, self.layer_settings = choose_base_model(
            model_name, dream_style, layer_settings
        )

        if isinstance(self.layer_settings, list):
            # A list of layers
            outputs = [
                self.model.get_layer(name).output for name in self.layer_settings
            ]
        else:
            # A dict of layers and its activation coefficients
            outputs = {
                layer.name: layer.output
                for layer in [
                    self.model.get_layer(name) for name in self.layer_settings.keys()
                ]
            }
        self.feature_extractor = Model(inputs=self.model.inputs, outputs=outputs)

    @tf.function
    def __call__(self, input):
        return self.feature_extractor(input)
