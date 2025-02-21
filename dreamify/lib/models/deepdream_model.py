import tensorflow as tf


class DeepDreamModel(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
        )
    )
    def __call__(self, img, learning_rate):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = DeepDreamModel.calc_loss(img, self.model)
        gradients = tape.gradient(loss, img)
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        img = img + gradients * learning_rate
        img = tf.clip_by_value(img, -1, 1)
        return loss, img

    def gradient_ascent_loop(self, img, iterations, learning_rate, config):
        print("Tracing DeepDream computation graph...")
        loss = tf.constant(0.0)
        for _ in tf.range(iterations):
            loss, img = self.__call__(img, learning_rate)

            framer = config.framer

            if config.enable_framing and framer.continue_framing():
                framer.add_to_frames(img)

        return loss, img

    @staticmethod
    def calc_loss(cls, img, model):
        """Calculate the DeepDream loss by maximizing activations."""
        img_batch = tf.expand_dims(img, axis=0)
        layer_activations = model(img_batch)
        if len(layer_activations) == 1:
            layer_activations = [layer_activations]

        return tf.reduce_sum([tf.math.reduce_mean(act) for act in layer_activations])
