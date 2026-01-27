"""Utily function to create a bounding box from city coordinates."""
from pyproj import Transformer

def create_bbox_from_city_coordinates(
        lat: float,
        lon: float,
        width: float = 10,
        height: float = 10) -> dict[str, float]:
    """Creates a bounding box (west, south, east, north) in lat/lon
    given the center coordinates, image size and pixel size.

    Args:
        lat (float): latitude of the center point.
        lon (float): longitude of the center point.
        width (float): width of the bounding box in kilometers.
        height (float): height of the bounding box in kilometers.

    Returns:
        dict[str, float]: bounding box coordinates (west, south, east, north).
    """
    half_size_m_x = (width * 1000) / 2
    half_size_m_y = (height * 1000) / 2

    # reprojection WGS84 -> Lambert 93
    to_l93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    to_wgs84 = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    cx, cy = to_l93.transform(lon, lat)
    minx, miny = cx - half_size_m_x, cy - half_size_m_y
    maxx, maxy = cx + half_size_m_x, cy + half_size_m_y

    west, south = to_wgs84.transform(minx, miny)
    east, north = to_wgs84.transform(maxx, maxy)

    return {"west": west, "south": south, "east": east, "north": north}
