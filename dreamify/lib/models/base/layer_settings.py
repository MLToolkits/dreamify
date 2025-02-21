from enum import Enum


def generate_shallow_settings(layers):
    """Assigns a consistent coefficient scale to a list of layers."""
    return {layer: coef for layer, coef in zip(layers, [1.0, 1.5, 2.0, 2.5])}


class ModelLayerSettings(Enum):
    INCEPTION_V3 = (
        ["mixed3", "mixed5"],
        generate_shallow_settings(["mixed4", "mixed5", "mixed6", "mixed7"]),
    )
    VGG19 = (
        ["block5_conv3", "block5_conv2"],
        generate_shallow_settings(
            ["block5_conv3", "block5_conv2", "block4_conv3", "block3_conv3"]
        ),
    )
    DENSENET121 = (
        ["conv5_block16_1_conv", "conv4_block24_1_conv"],
        generate_shallow_settings(
            [
                "conv5_block16_1_conv",
                "conv4_block24_1_conv",
                "conv3_block16_1_conv",
                "conv2_block12_1_conv",
            ]
        ),
    )
    EFFICIENTNET_V2L = (
        ["block7a_project_bn", "block6a_expand_activation"],
        generate_shallow_settings(
            [
                "block7a_project_bn",
                "block6a_expand_activation",
                "block5a_expand_activation",
                "block4a_expand_activation",
            ]
        ),
    )
    INCEPTION_RESNET_V2 = (
        ["mixed_7a", "mixed_6a"],
        generate_shallow_settings(["mixed_7a", "mixed_6a", "mixed_5a", "mixed_4a"]),
    )
    RESNET152V2 = (
        ["conv5_block3_out", "conv4_block6_out"],
        generate_shallow_settings(
            [
                "conv5_block3_out",
                "conv4_block6_out",
                "conv3_block4_out",
                "conv2_block3_out",
            ]
        ),
    )
    XCEPTION = (
        ["block14_sepconv2_act", "block13_sepconv2_act"],
        generate_shallow_settings(
            [
                "block14_sepconv2_act",
                "block13_sepconv2_act",
                "block12_sepconv2_act",
                "block11_sepconv2_act",
            ]
        ),
    )
    CONVNEXT_XL = (
        ["stage4_block2_depthwise_conv", "stage3_block2_depthwise_conv"],
        generate_shallow_settings(
            [
                "stage4_block2_depthwise_conv",
                "stage3_block2_depthwise_conv",
                "stage2_block2_depthwise_conv",
                "stage1_block2_depthwise_conv",
            ]
        ),
    )
    MOBILENET_V2 = (
        ["block_16_depthwise", "block_13_depthwise"],
        generate_shallow_settings(
            [
                "block_16_depthwise",
                "block_13_depthwise",
                "block_8_depthwise",
                "block_5_depthwise",
            ]
        ),
    )

    @property
    def deep(self):
        """Returns deep dream layers."""
        return self.value[0]

    @property
    def shallow(self):
        """Returns shallow dream layers with activation coefficients."""
        return self.value[1]
