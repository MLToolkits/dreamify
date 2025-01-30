from dataclasses import dataclass


@dataclass
class Config:
    feature_extractor: object = None
    layer_settings: object = None
    original_shape: object = None
    enable_framing: bool = False
    frames_for_vid: list = None
    max_frames_to_sample: int = 0
    curr_frame_idx: int = 0
    upsample_vid: bool = False
