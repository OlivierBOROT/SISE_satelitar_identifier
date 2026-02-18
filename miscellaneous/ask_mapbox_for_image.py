from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


def ask_mapbox_for_image(
    image_width_height: dict[str, int],
    mapbox_token: str,
    bounding_box: dict[str, float],
    request_timeout: int = 10,
    output_file: Path | bool = False,
) -> Image.Image:
    """Ask Mapbox for a satellite image of a given bounding box.

    Args:
        image_width_height (dict[str, int]): Dictionary with keys 'width' and 'height' for image dimensions.
        mapbox_token (str): Mapbox access token.
        bounding_box (dict[str, float]): Dictionary with keys 'west', 'south', 'east', 'north' for bounding box coordinates.
        request_timeout (int, optional): Timeout for the Mapbox API request in seconds. Defaults to 10 seconds.
        output_file (Path | bool, optional): Path to save the output image. If False, no file is saved. Defaults to False.

    Raises:
        ValueError: If bounding box coordinates are not provided.
        ValueError: If Mapbox access token is not provided.
        ValueError: If image width or height is not provided.
    """
    west = bounding_box.get("west", None)
    south = bounding_box.get("south", None)
    east = bounding_box.get("east", None)
    north = bounding_box.get("north", None)

    if None in [west, south, east, north]:
        raise ValueError("Bounding box coordinates must be provided.")

    if not mapbox_token:
        raise ValueError("Mapbox access token is required.")

    image_width = image_width_height.get("width", None)
    image_height = image_width_height.get("height", None)

    if None in [image_width, image_height]:
        raise ValueError("Image width and height must be provided.")

    img_url: str = (
        "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
        f"[{west},{south},{east},{north}]/{image_width}x{image_height}"
        f"?access_token={mapbox_token}"
    )

    img = requests.get(img_url, timeout=request_timeout).content
    if output_file:
        image = Image.open(BytesIO(img))
        image.save(output_file)
    return image
