# import os
# import tempfile

# import tensorflow as tf
# from moviepy import VideoFileClip
# from moviepy.video.compositing import CompositeVideoClip
# from moviepy.video.fx import AccelDecel, TimeSymmetrize
# from moviepy.video.VideoClip import DataVideoClip

# from dreamify.utils.common import deprocess


# class ImageToVideoConverterHybrid:
#     def __init__(self, dimensions, max_frames_to_sample, duration):
#         self.dimensions = dimensions
#         self.current_chunk: list = []
#         self.max_frames_to_sample: int = max_frames_to_sample
#         self.curr_frame_idx: int = 0
#         self.num_frames_to_insert: int = 0
#         self.duration: int = duration

#         self.FPS: int = 30
#         self.MAX_FRAMES_IN_MEM: int = 50

#         self.chunk_files: list = []
#         self.temp_folder = tempfile.mkdtemp()
#         print(f"Temporary folder created at {self.temp_folder}")

#     def add_to_frames(self, frame):
#         frame = tf.image.resize(frame, self.dimensions)
#         frame = deprocess(frame)
#         self.current_chunk.append(frame.numpy())
#         self.curr_frame_idx += 1

#         if len(self.current_chunk) >= self.MAX_FRAMES_IN_MEM:
#             self.flush_chunk()

#     def continue_framing(self):
#         return self.curr_frame_idx < self.max_frames_to_sample - 1

#     def flush_chunk(self):
#         if not self.current_chunk:
#             return

#         chunk_frames = []
#         for i in range(len(self.current_chunk) - 1):
#             chunk_frames.append(self.current_chunk[i])
#             interpolated = self.interpolate_frames(
#                 tf.cast(self.current_chunk[i], tf.float32),
#                 tf.cast(self.current_chunk[i + 1], tf.float32),
#                 30,
#             )
#             chunk_frames.extend(interpolated)

#         chunk_frames.append(self.current_chunk[-1])

#         clip = DataVideoClip(self.current_chunk, lambda x: x, fps=self.FPS)
#         chunk_path = os.path.join(
#             self.temp_folder, f"chunk_{len(self.chunk_files)}.mp4"
#         )
#         clip.write_videofile(chunk_path, logger=None)
#         self.chunk_files.append(chunk_path)
#         self.current_chunk = []

#     def to_video(
#         self,
#         output_path="dream.mp4",
#         duration=3,
#         mirror_video=False,
#     ):
#         self.duration = duration
#         self.flush_chunk()
#         clips = [VideoFileClip(chunk) for chunk in self.chunk_files]
#         final_clip = CompositeVideoClip.concatenate_videoclips(clips)
#         if mirror_video:
#             final_clip = TimeSymmetrize().apply(final_clip)
#         final_clip = AccelDecel(new_duration=duration).apply(final_clip)
#         final_clip.write_videofile(output_path, logger=None)

#     def interpolate_frames(self, frame1, frame2, num_frames):
#         alphas = tf.linspace(0.0, 1.0, num_frames + 2)[1:-1]
#         interpolated_frames = (1 - alphas[:, None, None, None]) * frame1 + alphas[
#             :, None, None, None
#         ] * frame2
#         return tf.cast(interpolated_frames, tf.uint8)

#     def calculate_num_frames_to_insert(self):
#         """
#         Calculate the number of frames to interpolate to ensure video smoothness of 30fps

#         Derivation:
#                 30 = (max_frames_to_sample * num_frames_to_insert) // duration
#              => 30 * duration = max_frames_to_sample * num_frames_to_insert
#              => 30 * duration // max_frames_to_sample = num_frames_to_insert
#              ≡ num_frames_to_insert = (30 * duration) // max_frames_to_sample
#         """
#         return (self.FPS * self.duration) // self.max_frames_to_sample
