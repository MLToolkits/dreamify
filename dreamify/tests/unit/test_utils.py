from dreamify.utils.common import get_image, show


def test_get_image_by_URL():
    url = (
        "https://storage.googleapis.com/download.tensorflow.org/"
        "example_images/YellowLabradorLooking_new.jpg"
    )
    img = get_image(url, max_dim=500)
    show(img)

    img = get_image(url)
    show(img)


def test_get_image_by_path():
    path = "examples/example0.jpg"
    img = get_image(path, max_dim=500)
    show(img)

    img = get_image(path)
    show(img)
