import tensorflow as tf
from tensorflow.keras import Model

from dreamify.lib.models.base import choose_base_model


class FeatureExtractor:
    def __init__(self, model_name, dream_style, layer_settings):
        self.model, self.layer_settings = choose_base_model(
            model_name, dream_style, layer_settings
        )

        # outputs is either:
        # - A list of layer outputs (for DeepDream)
        # - A dictionary of layer outputs with activation coefficients (for Dream)
        outputs = (
            [self.model.get_layer(name).output for name in self.layer_settings]
            if isinstance(self.layer_settings, list)
            else {
                name: self.model.get_layer(name).output for name in self.layer_settings
            }
        )

        self.feature_extractor = Model(inputs=self.model.inputs, outputs=outputs)

    @tf.function
    def __call__(self, input):
        return self.feature_extractor(input)
