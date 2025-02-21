import tensorflow as tf


class TiledGradients(tf.Module):
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[2], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
        )
    )
    def __call__(self, img, img_size, tile_size=512):
        shift, img_rolled = TiledGradients.random_roll(img, tile_size)

        # Initialize the image gradients to zero.
        gradients = tf.zeros_like(img_rolled)

        # Skip the last tile, unless there's only one tile.
        xs = tf.range(0, img_size[1], tile_size)[:-1]
        if not tf.cast(len(xs), bool):
            xs = tf.constant([0])
        ys = tf.range(0, img_size[0], tile_size)[:-1]
        if not tf.cast(len(ys), bool):
            ys = tf.constant([0])

        for x in xs:
            for y in ys:
                # Calculate the gradients for this tile.
                with tf.GradientTape() as tape:
                    # This needs gradients relative to `img_rolled`.
                    # `GradientTape` only watches `tf.Variable`s by default.
                    tape.watch(img_rolled)

                    # Extract a tile out of the image.
                    img_tile = img_rolled[y : y + tile_size, x : x + tile_size]
                    loss = TiledGradients.calc_loss(img_tile, self.feature_extractor)

                # Update the image gradients for this tile.
                gradients = gradients + tape.gradient(loss, img_rolled)

        # Undo the random shift applied to the image and its gradients.
        gradients = tf.roll(gradients, shift=-shift, axis=[0, 1])

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8

        return gradients

    @staticmethod
    def calc_loss(img, feature_extractor):
        """Calculate the DeepDream loss by maximizing activations."""
        img_batch = tf.expand_dims(img, axis=0)
        layer_activations = feature_extractor(img_batch)
        if len(layer_activations) == 1:
            layer_activations = [layer_activations]

        return tf.reduce_sum([tf.math.reduce_mean(act) for act in layer_activations])

    @staticmethod
    def random_roll(img, maxroll):
        # Randomly shift the image to avoid tiled boundaries.
        shift = tf.random.uniform(
            shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32
        )
        img_rolled = tf.roll(img, shift=shift, axis=[0, 1])
        return shift, img_rolled


__all__ = [TiledGradients]
