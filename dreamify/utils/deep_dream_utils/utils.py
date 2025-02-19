import tensorflow as tf


def calc_loss(img, model):
    """Calculate the DeepDream loss by maximizing activations."""
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    return tf.reduce_sum([tf.math.reduce_mean(act) for act in layer_activations])
