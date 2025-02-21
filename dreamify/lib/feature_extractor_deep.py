import tensorflow as tf
from tensorflow import keras

from dreamify.lib.deep_models import choose_deep_model


class FeatureExtractorDeep:
    def __init__(self, model_name, dream_style, layer_settings):
        self.model, self.layers = choose_deep_model(
            model_name, dream_style, layer_settings
        )

        self.feature_extractor = keras.Model(
            inputs=self.model.inputs, outputs=self.layers
        )

    @tf.function
    def __call__(self, input):
        return self.feature_extractor(input)
