# Dreamify

A function that applies deep dream to an image using pre-trained CNNs trained on the ImageNet dataset.

## Installation

``` bash
pip install dreamify
```

## Usage

To apply Dreamify to an image, use the following Python script:

```python
from dreamify import generate_dream_image


image_path = "example.jpg"

generate_dream_image(image_path):
```

You may customize the behavior of the dreamifyer by selecting a different pre-trained model, saving it as a video, etc.:

```python
from dreamify import generate_dream_image


image_path = "example.jpg"

generate_dream_image(
    image_path,
    output_path="dream.png",
    model_name="xception",
    step=20.0,
    num_octave=3,
    octave_scale=1.4,
    iterations=30,
    max_loss=15.0,
    save_video=False,
    duration=10,
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
| Inception-V3          | `inception_v3`         |
| ResNet152V2           | `resnet152v2`          |
| Xception (Default)              | `xception`             |
| MobileNet-V2          | `mobilenet_v2`         |


<p align="center">
  <img src="examples/cat-optimized.gif" alt="Cat" width="100%" />
</p>


<p align="center" width="100%">
  <img src="examples/example0.jpg" width="49.5%" style="margin-right: 10px;" />
  <img src="examples/dream0.png" width="49.5%" />
</p>

<p align="center">
  <img src="examples/example3.jpg" width="49.5%" style="margin-right: 10px;" />
  <img src="examples/dream3.png" width="49.5%" />
</p>



