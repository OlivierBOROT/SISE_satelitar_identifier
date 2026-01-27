"""Utility to create a bounding box from center coordinates, image size and pixel size."""
from pyproj import Transformer

def create_bbox_from_coordinates(
        lat: float,
        lon: float,
        img_height: int = 512 + 30,
        img_width: int = 512,
        pixel_size: float = 0.4) -> dict[str, float]:
    """Creates a bounding box (west, south, east, north) in lat/lon
    given the center coordinates, image size and pixel size.

    Args:
        lat (float): latitude of the center point.
        lon (float): longitude of the center point.
        img_height (int): height of the image in pixels. Default is 512px + 30px watermark.
        img_width (int): width of the image in pixels. Default is 512px.
        pixel_size (float): size of a pixel in meters. Default is 0.4m.

    Returns:
        dict[str, float]: bounding box coordinates (west, south, east, north).
    """
    half_size_m_x = (img_width * pixel_size) / 2
    half_size_m_y = (img_height * pixel_size) / 2

    # reprojection WGS84 -> Lambert 93
    to_l93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    to_wgs84 = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    cx, cy = to_l93.transform(lon, lat)
    minx, miny = cx - half_size_m_x, cy - half_size_m_y
    maxx, maxy = cx + half_size_m_x, cy + half_size_m_y

    west, south = to_wgs84.transform(minx, miny)
    east, north = to_wgs84.transform(maxx, maxy)

    return {"west": west, "south": south, "east": east, "north": north}
