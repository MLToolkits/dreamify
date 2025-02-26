import os
import tempfile

import psutil
import tensorflow as tf
from moviepy import VideoFileClip

# from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing import CompositeVideoClip
from moviepy.video.fx import AccelDecel, TimeSymmetrize
from moviepy.video.VideoClip import DataVideoClip

from dreamify.utils.common import deprocess_image

os.environ["IMAGEIO_FFMPEG_LOG_LEVEL"] = "quiet"


class ImageToVideoConverter:
    def __init__(self, dimensions, max_frames_to_sample):
        self.dimensions = dimensions
        self.current_chunk: list = []
        self.max_frames_to_sample: int = max_frames_to_sample
        self.curr_frame_idx: int = 0

        self.FPS: int = 30
        self.MAX_FRAMES_IN_MEM: int = self.calculate_max_frames_to_cache()
        self.FRAMES_TO_LERP = (
            10 if self.dimensions[0] * self.dimensions[1] >= 921_600 else 30
        )  # 1980 x 720 = 921,600 image is 'standard', as they say

        self.chunk_files: list = []
        self.temp_folder = tempfile.mkdtemp(prefix="buffer_")
        print(f"Temporary folder created at {self.temp_folder}")

    def calculate_max_frames_to_cache(self):
        """Calculate the number of frames that can be stored in memory."""
        h, w = self.dimensions
        frame_size = h * w * 3

        available_memory = psutil.virtual_memory().available * 0.6

        max_frames = available_memory // frame_size

        return max(10, min(50, int(max_frames)))

    def add_to_frames(self, frame):
        frame = tf.image.resize(frame, self.dimensions)
        frame = deprocess_image(frame)
        self.current_chunk.append(frame)
        self.curr_frame_idx += 1

        if len(self.current_chunk) >= self.MAX_FRAMES_IN_MEM:
            self.flush_chunk()

    def continue_framing(self):
        return self.curr_frame_idx < self.max_frames_to_sample - 1

    def flush_chunk(self):
        if not self.current_chunk:
            return

        chunk_frames = []
        for i in range(len(self.current_chunk) - 1):
            chunk_frames.append(self.current_chunk[i])
            interpolated = self.interpolate_frames(
                tf.cast(self.current_chunk[i], tf.float16),
                tf.cast(self.current_chunk[i + 1], tf.float16),
                tf.constant(self.FRAMES_TO_LERP),
            )
            chunk_frames.extend(interpolated)

        chunk_frames.append(self.current_chunk[-1])

        clip = DataVideoClip(self.current_chunk, lambda x: x, fps=self.FPS)
        chunk_path = os.path.join(
            self.temp_folder, f"chunk_{len(self.chunk_files)}.mp4"
        )
        clip.write_videofile(
            chunk_path,
            logger=None,
            ffmpeg_params=["-hide_banner", "-loglevel", "quiet", "-nostats"],
        )

        clip.close()

        self.chunk_files.append(chunk_path)
        self.current_chunk = []

    def to_video(
        self,
        output_path="dream.mp4",
        duration=3,
        mirror_video=False,
    ):
        self.flush_chunk()

        clips = [VideoFileClip(chunk) for chunk in self.chunk_files]
        final_clip = CompositeVideoClip.concatenate_videoclips(clips)

        final_clip = AccelDecel(new_duration=duration).apply(final_clip)

        if mirror_video:
            final_clip = TimeSymmetrize().apply(final_clip)

        # final_clip.with_audio(AudioFileClip("dreamify/assets/flight.wav"))

        final_clip.write_videofile(
            output_path,
            logger=None,
            ffmpeg_params=["-hide_banner", "-loglevel", "quiet", "-nostats"],
        )
        final_clip.close()

    def to_gif(
        self,
        output_path="dream.gif",
        duration=3,
        mirror_video=False,
    ):
        self.flush_chunk()

        clips = [VideoFileClip(chunk) for chunk in self.chunk_files]

        final_clip = CompositeVideoClip.concatenate_videoclips(clips)

        final_clip = AccelDecel(new_duration=duration).apply(final_clip)

        if mirror_video:
            final_clip = TimeSymmetrize().apply(final_clip)

        final_clip.write_gif(output_path, fps=30, logger=None)
        final_clip.close()

    @tf.function
    def interpolate_frames(self, frame1, frame2, num_frames):
        alphas = tf.cast(tf.linspace(0.0, 1.0, num_frames + 2)[1:-1], tf.float16)
        interpolated_frames = (1 - alphas[:, None, None, None]) * frame1 + alphas[
            :, None, None, None
        ] * frame2
        return tf.cast(interpolated_frames, tf.uint8)
