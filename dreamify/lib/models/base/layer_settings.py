from enum import Enum


class ShallowDreamModelLayerSettings(Enum):
    INCEPTION_V3 = {
        "mixed4": 1.0,
        "mixed5": 1.5,
        "mixed6": 2.0,
        "mixed7": 2.5,
    }
    VGG19 = {
        "block5_conv3": 1.0,
        "block5_conv2": 1.5,
        "block4_conv3": 2.0,
        "block3_conv3": 2.5,
    }
    DENSENET121 = {
        "conv5_block16_1_conv": 1.0,
        "conv4_block24_1_conv": 1.5,
        "conv3_block16_1_conv": 2.0,
        "conv2_block12_1_conv": 2.5,
    }
    EFFICIENTNET_V2L = {
        "block7a_project_bn": 1.0,
        "block6a_expand_activation": 1.5,
        "block5a_expand_activation": 2.0,
        "block4a_expand_activation": 2.5,
    }
    INCEPTION_RESNET_V2 = {
        "mixed_7a": 1.0,
        "mixed_6a": 1.5,
        "mixed_5a": 2.0,
        "mixed_4a": 2.5,
    }
    RESNET152V2 = {
        "conv5_block3_out": 1.0,
        "conv4_block6_out": 1.5,
        "conv3_block4_out": 2.0,
        "conv2_block3_out": 2.5,
    }
    XCEPTION = {
        "block14_sepconv2_act": 1.0,
        "block13_sepconv2_act": 1.5,
        "block12_sepconv2_act": 2.0,
        "block11_sepconv2_act": 2.5,
    }
    CONVNEXT_XL = {
        "stage4_block2_depthwise_conv": 1.0,
        "stage3_block2_depthwise_conv": 1.5,
        "stage2_block2_depthwise_conv": 2.0,
        "stage1_block2_depthwise_conv": 2.5,
    }
    MOBILENET_V2 = {
        "block_16_depthwise": 1.0,
        "block_13_depthwise": 1.5,
        "block_8_depthwise": 2.0,
        "block_5_depthwise": 2.5,
    }


class DeepDreamModelLayerSettings(Enum):
    INCEPTION_V3 = ["mixed3", "mixed5"]
    VGG19 = ["block5_conv3", "block5_conv2"]
    DENSENET121 = ["conv5_block16_1_conv", "conv4_block24_1_conv"]
    EFFICIENTNET_V2L = ["block7a_project_bn", "block6a_expand_activation"]
    INCEPTION_RESNET_V2 = ["mixed_7a", "mixed_6a"]
    RESNET152V2 = ["conv5_block3_out", "conv4_block6_out"]
    XCEPTION = ["block14_sepconv2_act", "block13_sepconv2_act"]
    CONVNEXT_XL = ["stage4_block2_depthwise_conv", "stage3_block2_depthwise_conv"]
    MOBILENET_V2 = ["block_16_depthwise", "block_13_depthwise"]
