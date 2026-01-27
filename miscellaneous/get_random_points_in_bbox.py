"""Generates random coordinates within a specified bounding box."""
import random

def get_random_points_in_bbox(
        bbox: dict[str, float],
        number_of_points: int = 10
    ) -> list[dict[str, float]]:
    """Generates a random list of coordinates (latitude, longitude) within a given bounding box.

    Args:
        bbox (dict[str, float]): Bounding box with keys "south", "north", "west", and "east".
        number_of_points (int, optional): Number of coordinates to generate. Defaults to 10.

    Returns:
        list[dict[str, float]]: List of generated coordinates (latitude, longitude).
    """
    coords: list[dict[str, float]] = []
    lat_min, lat_max = bbox["south"], bbox["north"]
    lon_min, lon_max = bbox["west"], bbox["east"]

    def generate_points() -> None:
            # Generate coordinates
        coords.extend([
            {"lat": random.uniform(lat_min, lat_max), "lon": random.uniform(lon_min, lon_max)}
            for _ in range(number_of_points)
        ])

    generate_points()
    # verify no duplicates
    while len(coords) != len(set(coords)):
        generate_points()

    return coords
