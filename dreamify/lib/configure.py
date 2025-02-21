from dataclasses import dataclass

from dreamify.lib.misc import ImageToVideoConverter


@dataclass
class Config:
    feature_extractor: object = None
    layer_settings: object = None
    original_shape: object = None
    save_video: bool = False
    save_gif: bool = False
    enable_framing: bool = False
    max_frames_to_sample: int = 0
    duration: int = 0
    vid_duration: int = 0
    gif_duration: int = 0
    mirror_video: bool = False
    framer: ImageToVideoConverter = None

    def __post_init__(self):
        if self.framer is None:
            self.framer = ImageToVideoConverter(
                dimensions=self.original_shape,
                max_frames_to_sample=self.max_frames_to_sample,
            )

        # Set `self.duration` based on `vid_duration` and `gif_duration` if `duration` is not provided
        if self.duration == 0:
            if self.vid_duration != 0:
                self.duration = self.vid_duration
            elif self.gif_duration != 0:
                self.duration = self.gif_duration

        # Set `vid_duration` and `gif_duration` to `self.duration` if they are not provided
        if self.vid_duration == 0 and self.duration != 0:
            self.vid_duration = self.duration
        if self.gif_duration == 0 and self.duration != 0:
            self.gif_duration = self.duration

    def __hash__(self):
        return hash("config")
