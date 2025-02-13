import tensorflow as tf
from moviepy.video.fx import AccelDecel, TimeSymmetrize
from moviepy.video.VideoClip import DataVideoClip

from dreamify.utils.common import deprocess


class ImageToVideoConverter:
    def __init__(self, dimensions, max_frames_to_sample):
        self.dimensions = dimensions
        self.frames_for_vid: list = []
        self.max_frames_to_sample: int = max_frames_to_sample
        self.curr_frame_idx: int = 0

    def add_to_frames(self, frame):
        frame = tf.image.resize(frame, self.dimensions)
        frame = deprocess(frame)

        self.frames_for_vid.append(frame)
        self.curr_frame_idx += 1

    def continue_framing(self):
        return self.curr_frame_idx < self.max_frames_to_sample - 1

    def to_video(self, output_path, duration, mirror_video, fps=60):
        self.upsample()

        frames = [frame.numpy() for frame in self.frames_for_vid]
        print(f"Number of images to frame: {len(frames)}")

        vid = DataVideoClip(frames, lambda x: x, fps=fps)
        if mirror_video:
            vid = TimeSymmetrize().apply(vid)
        vid = AccelDecel(new_duration=duration).apply(vid)
        vid.write_videofile(output_path)

    def upsample(self):
        NUM_FRAMES_TO_INSERT = 30

        new_frames = []

        # Upsample via frame-frame interpolation
        for i in range(len(self.frames_for_vid) - 1):
            frame1 = tf.cast(self.frames_for_vid[i], tf.float32)
            frame2 = tf.cast(self.frames_for_vid[i + 1], tf.float32)

            # Add original frame
            new_frames.append(self.frames_for_vid[i])

            interpolated = self.interpolate_frames(frame1, frame2, NUM_FRAMES_TO_INSERT)
            new_frames.extend(interpolated)

        new_frames.extend(
            [self.frames_for_vid[-1]] * 60 * 3
        )  # Lengthen end frame by 3 frames
        self.frames_for_vid = new_frames

    def interpolate_frames(self, frame1, frame2, num_frames):
        alphas = tf.linspace(0.0, 1.0, num_frames + 2)[1:-1]  # Avoid frames 0 and 1

        frame1 = tf.cast(frame1, tf.float32)
        frame2 = tf.cast(frame2, tf.float32)

        interpolated_frames = (1 - alphas[:, None, None, None]) * frame1 + alphas[
            :, None, None, None
        ] * frame2
        return tf.cast(interpolated_frames, tf.uint8)
