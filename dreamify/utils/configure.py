# from dataclasses import dataclass, field


# @dataclass
# class Config:
#     feature_extractor: object = None
#     layer_settings: object = None
#     original_shape: object = None
#     enable_framing: bool = False
#     frames_for_vid: list = field(default_factory=list)
#     max_frames_to_sample: int = 0
#     curr_frame_idx: int = 0


########################################################################
from dataclasses import dataclass

from dreamify.lib.image_to_video_converter import ImageToVideoConverter


@dataclass
class Config:
    feature_extractor: object = None
    layer_settings: object = None
    original_shape: object = None
    save_video: bool = False
    enable_framing: bool = False
    max_frames_to_sample: int = 0
    framer: ImageToVideoConverter = None

    def __post_init__(self):
        if self.framer is None:
            self.framer = ImageToVideoConverter(
                dimensions=self.original_shape,
                max_frames_to_sample=self.max_frames_to_sample,
            )


class ConfigSingleton:
    _config_instance: Config = None

    @classmethod
    def get_config(cls, **kwargs):
        if cls._config_instance is None:
            cls._config_instance = Config(**kwargs)
        return cls._config_instance
