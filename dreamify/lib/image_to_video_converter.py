import os
import tempfile

import imageio
import numpy as np
import tensorflow as tf
from moviepy.video.fx import AccelDecel, TimeSymmetrize
from moviepy.video.VideoClip import DataVideoClip

from dreamify.utils.common import deprocess


class ImageToVideoConverter:
    # TODO (PROPERLY RESET THE FRAME INDECES)
    def __init__(self, dimensions, max_frames_to_sample):
        self.dimensions = dimensions
        self.max_frames_to_sample = max_frames_to_sample
        self.curr_frame_idx = 0
        self.total_frames = 0
        self.NUM_FRAMES_TO_INSERT = 15
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

    def to_video(self, output_path, duration, mirror_video, fps=60):
        self.upsample(fps)

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Read buffered frames from disk
        frames = [
            imageio.imread(os.path.join(self.temp_folder, f"frame_{i:04d}.png"))
            for i in range(self.total_frames)  # Use total_frames here
        ]
        print(f"Number of images to frame: {len(frames)}")

        # Create the video
        vid = DataVideoClip(frames, lambda x: x, fps=fps)
        if mirror_video:
            vid = TimeSymmetrize().apply(vid)
        vid = AccelDecel(new_duration=duration).apply(vid)
        vid.write_videofile(output_path)

        # Clean up ops
        self.curr_frame_idx = 0
        self.total_frames = 0
        self.cleanup_temp_folder()

    def upsample(self, fps):
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
                frame1, frame2, self.NUM_FRAMES_TO_INSERT
            )
            new_frames.extend(interpolated)

        new_frames.extend([frame2] * fps * 60)  # Lengthen end frame by 60 units
        # Save the upsampled frames back to disk
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
