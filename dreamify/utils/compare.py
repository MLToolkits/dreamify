from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from dreamify.dream import generate_dream_image
from dreamify.utils.models import ModelType


def main():
    image_path = "examples/example0.jpg"
    output_dir = Path("examples")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_images = []

    for model_name in ModelType:
        print(f"Generating dream image with model: {model_name.value}")

        output_path = output_dir / f"dream_{model_name.value}.png"

        generate_dream_image(
            image_path=image_path,
            output_path=str(output_path),
            model_name=model_name.value,
            step=20.0,
            num_octave=3,
            octave_scale=1.4,
            iterations=30,
            max_loss=15.0,
            save_video=False,
            duration=10,
        )
        generated_images.append(output_path)
        print(
            f"Dream image generated for {model_name.value} and saved to {output_path}"
        )

    compare_images(generated_images)


def compare_images(image_paths):
    fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 5))

    if len(image_paths) == 1:
        axes = [axes]

    for ax, image_path in zip(axes, image_paths):
        img = Image.open(image_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(image_path.name)

    plt.show()


if __name__ == "__main__":
    main()
