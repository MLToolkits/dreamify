import tensorflow as tf
from tensorflow import keras

from dreamify.utils.models import choose_model


class FeatureExtractor:
    def __init__(self, model_name):
        self.model, self.layer_settings = choose_model(model_name)

        outputs_dict = {
            layer.name: layer.output
            for layer in [
                self.model.get_layer(name) for name in self.layer_settings.keys()
            ]
        }
        self.feature_extractor = keras.Model(
            inputs=self.model.inputs, outputs=outputs_dict
        )

    @tf.function
    def __call__(self, input):
        return self.feature_extractor(input)
