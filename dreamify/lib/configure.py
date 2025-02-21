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
    mirror_video: bool = False
    framer: ImageToVideoConverter = None

    def __post_init__(self):
        if self.framer is None:
            self.framer = ImageToVideoConverter(
                dimensions=self.original_shape,
                max_frames_to_sample=self.max_frames_to_sample,
            )

    def __hash__(self):
        return hash("config")
