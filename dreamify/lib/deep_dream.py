# import tensorflow as tf

# from dreamify.utils.deep_dream_utils.utils import calc_loss


# class DeepDream(tf.Module):
#     def __init__(self, model, config):
#         self.model = model
#         self.config = config

#     @tf.function(
#         input_signature=(
#             tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
#             tf.TensorSpec(shape=[], dtype=tf.int32),
#             tf.TensorSpec(shape=[], dtype=tf.float32),
#         )
#     )
#     def __call__(self, img, steps, step_size):
#         print("Tracing DeepDream computation graph...")
#         loss = tf.constant(0.0)
#         for n in tf.range(steps):
#             with tf.GradientTape() as tape:
#                 tape.watch(img)
#                 loss = calc_loss(img, self.model)
#                 gradients = tape.gradient(loss, img)
#             gradients /= tf.math.reduce_std(gradients) + 1e-8
#             img = img + gradients * step_size
#             img = tf.clip_by_value(img, -1, 1)

#             # framer = self.config.framer

#             # if self.config.enable_framing and framer.continue_framing():
#             #     framer.add_to_frames(img)

#         return loss, img

import tensorflow as tf

from dreamify.utils.deep_dream_utils.utils import calc_loss


class DeepDream(tf.Module):
    def __init__(self, model, config):
        self.model = model
        self.config = config

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
        )
    )
    def __call__(self, img, step_size):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = calc_loss(img, self.model)
            gradients = tape.gradient(loss, img)
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        img = img + gradients * step_size
        img = tf.clip_by_value(img, -1, 1)
        return loss, img

    def gradient_ascent_loop(self, img, steps, step_size, config):
        print("Tracing DeepDream computation graph...")
        loss = tf.constant(0.0)
        for _ in tf.range(steps):
            loss, img = self.__call__(img, step_size)

            framer = config.framer

            print("Current Frame Index: ", framer.curr_frame_idx)
            print("Max Frames to Sample: ", framer.max_frames_to_sample)

            if config.enable_framing and framer.continue_framing():
                framer.add_to_frames(img)

        return loss, img
