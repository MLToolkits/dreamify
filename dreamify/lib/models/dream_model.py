import tensorflow as tf
from tqdm import trange


class DreamModel(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
        )
    )
    def __call__(self, image, learning_rate, feature_extractor):
        with tf.GradientTape() as tape:
            tape.watch(image)
            loss = DreamModel.compute_loss(image, feature_extractor)
        grads = tape.gradient(loss, image)
        grads = tf.math.l2_normalize(grads)
        image += learning_rate * grads
        image = tf.clip_by_value(image, -1, 1)
        return loss, image

    def gradient_ascent_loop(
        self, image, iterations, learning_rate, max_loss=None, config=None
    ):
        for i in trange(
            iterations, desc="Gradient Ascent", unit="step", ncols=75, mininterval=0.1
        ):
            loss, image = gradient_ascent_step(
                image, learning_rate, config.feature_extractor
            )

            if max_loss is not None and loss > max_loss:
                print(
                    f"\nTerminating early: Loss ({loss:.2f}) exceeded max_loss ({max_loss:.2f}).\n"
                )
                break

            framer = config.framer

            if config.enable_framing and framer.continue_framing():
                framer.add_to_frames(image)

        return image

    @staticmethod
    def compute_loss(input_image, feature_extractor):
        features = feature_extractor(input_image)
        loss = tf.zeros(shape=())
        for name in features.keys():
            coeff = feature_extractor.layer_settings[name]
            activation = features[name]
            loss += coeff * tf.reduce_mean(tf.square(activation[:, 2:-2, 2:-2, :]))
        return loss
