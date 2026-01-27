from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


def ask_mapbox_for_image(
    output_file: Path,
    image_width_height: dict[str, int],
    mapbox_token: str,
    bounding_box: dict[str, float],
    request_timeout: int = 10,
) -> Image.Image:
    """Ask Mapbox for a satellite image of a given bounding box.

    Args:
        output_file (Path): Path to save the output image.
        image_width_height (dict[str, int]): Dictionary with keys 'width' and 'height' for image dimensions.
        mapbox_token (str): Mapbox access token.
        west (float, optional): Western boundary of the bounding box. Defaults to None.
        south (float, optional): Southern boundary of the bounding box. Defaults to None.
        east (float, optional): Eastern boundary of the bounding box. Defaults to None.
        north (float, optional): Northern boundary of the bounding box. Defaults to None.
        bounding_box (dict[str, float]):
        Dictionary containing bounding box coordinates.
        request_timeout (int, optional): Timeout for the HTTP request in seconds. Defaults to 10.

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

    if not output_file:
        output_file = Path.cwd() / "image_satellite.png"
        print(f"No output file provided, image will be saved at {output_file}")

    img_url: str = (
        "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
        f"[{west},{south},{east},{north}]/{image_width}x{image_height}"
        f"?access_token={mapbox_token}"
    )

    img = requests.get(img_url, timeout=request_timeout).content
    image = Image.open(BytesIO(img))
    image.save(output_file)
    return image
