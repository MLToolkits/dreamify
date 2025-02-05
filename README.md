# Dreamify

A function that applies deep dream to an image using pre-trained CNNs trained on the ImageNet dataset.

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
from dreamify.dream import generate_dream_image


image_path = "example.jpg"

generate_dream_image(image_path):
```

You may customize the behavior of the dreamifyer by selecting a different pre-trained model, saving it as a video, etc.:

```python
from dreamify.dream import generate_dream_image


image_path = "example.jpg"

generate_dream_image(
    image_path,
    output_path="dream.png",
    model_name="inception_v3",
    learning_rate=5.0,
    num_octave=5,
    octave_scale=1.3,
    iterations=100,
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
| Inception-V3 (Default)         | `inception_v3`         |
| ResNet152V2           | `resnet152v2`          |
| Xception               | `xception`             |
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



