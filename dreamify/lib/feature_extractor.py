import tensorflow as tf
from tensorflow import keras

from dreamify.lib.models import choose_base_model


class FeatureExtractor:
    def __init__(self, model_name, dream_style, layer_settings):
        self.model, self.layer_settings = choose_base_model(
            model_name, dream_style, layer_settings
        )

        if isinstance(self.layer_settings, list):
            # When layer_settings is a list, assume it contains layer names.
            outputs = [
                self.model.get_layer(name).output for name in self.layer_settings
            ]
        else:
            # Otherwise, assume layer_settings is a dict and its keys are layer names.
            outputs = {
                layer.name: layer.output
                for layer in [
                    self.model.get_layer(name) for name in self.layer_settings.keys()
                ]
            }
        self.feature_extractor = keras.Model(inputs=self.model.inputs, outputs=outputs)

    @tf.function
    def __call__(self, input):
        return self.feature_extractor(input)
