from pathlib import Path

import pytest

from dreamify.dream import generate_dream_image


@pytest.mark.filterwarnings(
    "ignore:The structure of `inputs` doesn't match the expected structure"
)
def test_mock_dream():
    img_path = "examples/example0.jpg"
    out_path = "examples/mock_dream.jpg"

    generate_dream_image(
        img_path, output_path=out_path, octaves=1, iterations=5, learning_rate=15.0
    )
    Path(out_path).unlink(missing_ok=True)


def test_dream_validator():
    img_path = "examples/example0.jpg"
    out_path = "examples/mock_dream.jpg"

    with pytest.raises(ValueError):
        generate_dream_image(
            img_path,
            output_path=out_path,
            num_octave=-1,
            iterations=5,
            learning_rate=15.0,
        )
        print("Handled negative")
