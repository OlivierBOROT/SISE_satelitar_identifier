"""Module for resizing images using the Pillow library."""

from PIL import Image


def crop_image(image: Image, top: int, left: int, right: int, bottom: int) -> Image:
    """
    Crop the given image to the specified bounding box.

    Args:
        image (PIL.Image): The image to be cropped.
        top (int): The top pixel coordinate of the bounding box.
        left (int): The left pixel coordinate of the bounding box.
        right (int): The right pixel coordinate of the bounding box.
        bottom (int): The bottom pixel coordinate of the bounding box.

    Returns:
        PIL.Image: The cropped image.
    """
    return image.crop((left, top, right, bottom))
