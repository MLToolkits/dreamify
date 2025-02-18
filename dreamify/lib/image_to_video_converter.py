import os
import tempfile

import imageio.v2 as imageio
import numpy as np
import tensorflow as tf
from moviepy.video.fx import AccelDecel, TimeSymmetrize
from moviepy.video.VideoClip import DataVideoClip

from dreamify.utils.common import deprocess


class ImageToVideoConverter:
    def __init__(self, dimensions, max_frames_to_sample):
        self.dimensions = dimensions
        self.max_frames_to_sample = max_frames_to_sample
        self.curr_frame_idx = 0
        self.total_frames = 0
        self.num_frames_to_insert: int = 0
        self.FPS: int = 30

        self.temp_folder = tempfile.mkdtemp()
        print(f"Temporary folder created at {self.temp_folder}")

    def add_to_frames(self, frame):
        frame = tf.image.resize(frame, self.dimensions)
        frame = deprocess(frame).numpy().astype("float32")

        # Buffering the frame to the disk
        frame_filename = os.path.join(
            self.temp_folder, f"frame_{self.curr_frame_idx:04d}.png"
        )
        imageio.imwrite(
            frame_filename, frame.astype("uint8")
        )  # Save the frame as an image
        self.curr_frame_idx += 1

    def continue_framing(self):
        return self.curr_frame_idx < self.max_frames_to_sample  # Use total_frames here

    def to_video(
        self,
        output_path="dream.mp4",
        duration=3,
        extend_ending=False,
        mirror_video=False,
    ):
        self.duration = duration
        self.num_frames_to_insert = self.calculate_num_frames_to_insert()

        self.upsample()

        output_dir = os.path.dirname(output_path)
        if output_dir != "" and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_path)

        # Read buffered frames from disk
        frames = [
            imageio.imread(os.path.join(self.temp_folder, f"frame_{i:04d}.png"))
            for i in range(self.total_frames)  # Use total_frames here
        ]
        print(f"Number of images to frame: {len(frames)}")

        # Create the video
        vid = DataVideoClip(frames, lambda x: x, fps=self.FPS)
        if mirror_video:
            vid = TimeSymmetrize().apply(vid)
        vid = AccelDecel(new_duration=duration).apply(vid)
        vid.write_videofile(output_path)

        # Clean up ops
        self.cleanup_temp_folder()

    def upsample(self):
        new_frames = []

        # Upsample via frame-frame interpolation
        for i in range(self.curr_frame_idx - 1):
            frame1 = imageio.imread(
                os.path.join(self.temp_folder, f"frame_{i:04d}.png")
            )
            frame2 = imageio.imread(
                os.path.join(self.temp_folder, f"frame_{i + 1:04d}.png")
            )

            # Add original frame
            new_frames.append(frame1)
            self.total_frames += 1  # Update total frames count

            interpolated = self.interpolate_frames(
                frame1, frame2, self.num_frames_to_insert
            )
            new_frames.extend(interpolated)

        self.save_upsampled_frames(new_frames)

    def interpolate_frames(self, frame1, frame2, num_frames):
        alphas = np.linspace(0.0, 1.0, num_frames + 2)[1:-1]  # Avoid frames 0 and 1

        interpolated_frames = (1 - alphas[:, None, None, None]) * frame1 + alphas[
            :, None, None, None
        ] * frame2
        return interpolated_frames.astype("uint8")

    def save_upsampled_frames(self, new_frames):
        for idx, frame in enumerate(new_frames):
            frame_filename = os.path.join(
                self.temp_folder, f"frame_{self.total_frames:04d}.png"
            )
            imageio.imwrite(frame_filename, frame)
            self.total_frames += 1  # Update total frames count

    def cleanup_temp_folder(self):
        # Delete all frames in the temporary folder
        for file_name in os.listdir(self.temp_folder):
            file_path = os.path.join(self.temp_folder, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(self.temp_folder)  # Remove the folder itself
        print(f"Temporary folder at {self.temp_folder} has been cleaned up")

    def calculate_num_frames_to_insert(self):
        """
        Calculate the number of frames to interpolate to ensure video smoothness of 30fps

        Derivation:
                30 = (max_frames_to_sample * num_frames_to_insert) // duration
             => 30 * duration = max_frames_to_sample * num_frames_to_insert
             => 30 * duration // max_frames_to_sample = num_frames_to_insert
             ≡ num_frames_to_insert = (30 * duration) // max_frames_to_sample
        """
        return (self.FPS * self.duration) // self.max_frames_to_sample

    def __hash__(self):
        return hash(self.name)
