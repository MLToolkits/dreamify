# Dreamify

A function that applies deep dream to an image using a pre-trained CNN trained on the ImageNet dataset.

<p align="center" width="100%">
  <img src="examples/doggy.gif" alt="Doggy" height="250px" />
  <img src="examples/cat-optimized.gif" alt="Cat" height="250px" />
</p>



## Testing it
``` bash
dreamify
```

## Installation

``` bash
pip install dreamify
```

## Usage

To apply Dreamify to an image, use the following Python script:

```python
from dreamify.deepdream import deepdream


image_path = "example.jpg"

deepdream(image_path)
```

You may customize the behavior of the dreamifyer by selecting a different pre-trained model, saving it as a video, etc.:

```python
from dreamify.deepdream import deepdream


image_path = "example.jpg"

deepdream(
    image_path,
    output_path="deepdream.png",
    model_name="inception_v3",
    iterations=100,
    learning_rate=0.01,
    octaves=range(-2, 3),
    octave_scale=1.3,
    save_video=False,
    save_gif=False,
    duration=3,
    vid_duration=3,
    gif_duration=3,
    mirror_video=False,
    seed=None,
)
```

## Available Models

Dreamify supports the following models:

| Model Name             | Enum Value              |
|------------------------|------------------------|
| VGG19                 | `vgg19`                |
| ConvNeXt-XL           | `convnext_xl`          |
| DenseNet121           | `densenet121`          |
| EfficientNet-V2L      | `efficientnet_v2l`     |
| Inception-ResNet-V2   | `inception_resnet_v2`  |
| Inception-V3 (Default)         | `inception_v3`         |
| ResNet152V2           | `resnet152v2`          |
| Xception               | `xception`             |
| MobileNet-V2          | `mobilenet_v2`         |

## Other Examples

###### DeepDream
<p align="center">
  <img src="examples/exampledeep.jpg" width="49.5%" style="margin-right: 10px;" />
  <img src="examples/exampledeepdream.png" width="49.5%" />
</p>
<p align="center">
  <img src="examples/exampledeep2.jpg" width="49.5%" style="margin-right: 10px;" />
  <img src="examples/exampledeepdream2.png" width="49.5%" />
</p>
<p align="center">
  <img src="examples/exampledeep1.jpg" width="49.5%" style="margin-right: 10px;" />
  <img src="examples/exampledeepdream1.png" width="49.5%" />
</p>
<p align="center">
  <img src="examples/grayscale.jpg" width="49.5%" style="margin-right: 10px;" />
  <img src="examples/deepdream_grayscale.png" width="49.5%" />
</p>

###### Dream (shallow) -- See documentation of [dream (shallow)](https://github.com/MLToolkits/dreamify/blob/3d3f3354b23d00a269427d2f50720874fa10f663/dreamify/dream.py#L26).
<p align="center" width="100%">
  <img src="examples/example0.jpg" width="49.5%" style="margin-right: 10px;" />
  <img src="examples/dream0.png" width="49.5%" />
</p>




