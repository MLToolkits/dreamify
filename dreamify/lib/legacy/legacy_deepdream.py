# @validate_dream
# def deep_dream_simple(
#     image_path=None,
#     img=None,
#     output_path="dream.png",
#     model_name="inception_v3",
#     learning_rate=0.01,
#     iterations=100,
#     save_video=False,
#     duration=3,
#     mirror_video=False,
#     config=None,
# ):
#     if image_path is None and img is None:
#         raise TypeError("Missing image_path or img argument")

#     base_image_path = Path(image_path)
#     output_path = Path(output_path)

#     base_model = tf.keras.applications.InceptionV3(
#         include_top=False, weights="imagenet"
#     )

#     names = ["mixed3", "mixed5"]
#     layers = [base_model.get_layer(name).output for name in names]

#     ft_ext = tf.keras.Model(inputs=base_model.input, outputs=layers)
#     dream_model = DeepDream(ft_ext)

#     img = get_image(base_image_path)
#     img = tf.keras.applications.inception_v3.preprocess_input(img)
#     img_shape = img.shape[1:3]

#     if config is None:
#         config = Config(
#             feature_extractor=dream_model,
#             layer_settings=dream_model.model.layers,
#             original_shape=img_shape,
#             save_video=save_video,
#             enable_framing=save_video,
#             max_frames_to_sample=iterations,
#         )

#     img = tf.keras.applications.inception_v3.preprocess_input(img)
#     img = tf.convert_to_tensor(img)

#     learning_rate = tf.convert_to_tensor(learning_rate)
#     iterations_remaining = iterations
#     iteration = 0
#     while iterations_remaining:
#         run_iterations = tf.constant(min(100, iterations_remaining))
#         iterations_remaining -= run_iterations
#         iteration += run_iterations

#         loss, img = dream_model.gradient_ascent_loop(
#             img, run_iterations, tf.constant(learning_rate), config
#         )

#         # display.clear_output(wait=True)
#         show(deprocess(img))
#         print("Iteration {}, loss {}".format(iteration, loss))

#     tf.keras.utils.save_img(output_path, img)
#     print(f"Dream image saved to {output_path}")

#     if save_video:
#         config.framer.to_video(output_path.stem + ".mp4", duration, mirror_video)

#     return img


# @validate_dream
# def deep_dream_octaved(
#     image_path,
#     output_path="dream.png",
#     model_name="inception_v3",
#     learning_rate=0.01,
#     iterations=100,
#     save_video=False,
#     duration=3,
#     mirror_video=False,
# ):
#     base_image_path = Path(image_path)
#     output_path = Path(output_path)

#     base_model = tf.keras.applications.InceptionV3(
#         include_top=False, weights="imagenet"
#     )

#     names = ["mixed3", "mixed5"]
#     layers = [base_model.get_layer(name).output for name in names]

#     ft_ext = tf.keras.Model(inputs=base_model.input, outputs=layers)
#     dream_model = DeepDream(ft_ext)

#     img = get_image(base_image_path)
#     img = tf.keras.applications.inception_v3.preprocess_input(img)
#     img_shape = img.shape[1:3]

#     config = Config(
#         feature_extractor=dream_model,
#         layer_settings=dream_model.model.layers,
#         original_shape=img_shape,
#         save_video=False,
#         enable_framing=save_video,
#         max_frames_to_sample=iterations * 5,  # 5 octaves
#     )

#     OCTAVE_SCALE = 1.30
#     img = tf.constant(np.array(img))
#     float_base_shape = tf.cast(tf.shape(img)[:-1], tf.float32)

#     for n in range(-2, 3):
#         if n == 2:
#             config.save_video = save_video

#         new_shape = tf.cast(float_base_shape * (OCTAVE_SCALE**n), tf.int32)
#         img = tf.image.resize(img, new_shape).numpy()
#         img = deep_dream_simple(
#             img=img,
#             model_name=model_name,
#             iterations=iterations,
#             learning_rate=learning_rate,
#             save_video=False,
#             duration=duration,
#             mirror_video=mirror_video,
#             config=config,
#         )

#     tf.keras.utils.save_img(output_path, img)
#     print(f"Dream image saved to {output_path}")

#     if save_video:
#         config.framer.to_video(output_path.stem + ".mp4", duration, mirror_video)

#     return img
