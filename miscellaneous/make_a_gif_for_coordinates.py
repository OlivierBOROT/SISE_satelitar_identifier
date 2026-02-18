"""Create an animated GIF showing building detection and bbox cleaning for given coordinates."""

from pathlib import Path
from typing import Tuple

import geopandas as gpd
import osmnx as ox
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import box

from miscellaneous.ask_mapbox_for_image import ask_mapbox_for_image
from miscellaneous.clean_overlapping_bboxes import clean_overlapping_bboxes
from miscellaneous.create_bbox_from_coordinates import create_bbox_from_coordinates


def _get_font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a TrueType font, fall back to default."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def _draw_geometries_on_image(
    base_image: Image.Image,
    gdf: gpd.GeoDataFrame,
    bbox: dict[str, float],
    outline_color: str = "red",
    line_width: int = 2,
    counter_text: str | None = None,
) -> Image.Image:
    """Draw GeoDataFrame geometries onto a copy of the base image.

    Args:
        base_image (Image.Image): Background satellite image.
        gdf (gpd.GeoDataFrame): GeoDataFrame with polygon geometries in EPSG:4326.
        bbox (dict[str, float]): Bounding box (west, south, east, north) of the image.
        outline_color (str): Color for the polygon outlines. Defaults to "red".
        line_width (int): Width of the outline. Defaults to 2.
        counter_text (str | None): Text to display in the top-right corner.

    Returns:
        Image.Image: Copy of the base image with geometries drawn on top.
    """
    img = base_image.copy()
    draw = ImageDraw.Draw(img)

    west, south = bbox["west"], bbox["south"]
    east, north = bbox["east"], bbox["north"]
    img_w, img_h = img.size

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]

        for poly in polys:
            coords = list(poly.exterior.coords)
            pixel_coords = [
                (
                    (lon - west) / (east - west) * img_w,
                    (north - lat) / (north - south) * img_h,
                )
                for lon, lat in coords
            ]
            draw.line(pixel_coords, fill=outline_color, width=line_width)

    # Draw counter in top-right corner
    if counter_text:
        font = _get_font(24)
        text_bbox = draw.textbbox((0, 0), counter_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        margin = 10
        x = img_w - text_w - margin
        y = margin
        # Draw background rectangle for readability
        draw.rectangle(
            [x - 5, y - 5, x + text_w + 5, y + text_h + 5],
            fill="black",
        )
        draw.text((x, y), counter_text, fill="white", font=font)

    return img


def make_a_gif_for_coordinates(
    coordinates: Tuple[float, float],
    output_file: str,
    mapbox_token: str,
    overlap_threshold: float = 0.3,
    max_cleaning_iterations: int = 50,
    frame_duration: int = 1000,
    loop: int = 0,
    image_cache_dir: str | Path | None = None,
) -> Path:
    """Create an animated GIF showing satellite image, building polygons, and progressive bbox cleaning.

    Steps:
        1. Create a bounding box for the given coordinates.
        2. Fetch a satellite image from Mapbox (with 30px extra for watermark).
        3. Crop the 30px watermark at the bottom.
        4. Add the satellite image as the first GIF frame.
        5. Fetch building polygons from OSM for the bounding box.
        6. Draw building bounding boxes on the image and add to the GIF.
        7. Use clean_overlapping_bboxes to clean the bboxes and add the result to the GIF.
        Repeat 6 & 7 for *cleaning_iterations* passes.

    Args:
        coordinates (Tuple[float, float]): (latitude, longitude) of the center point.
        output_file (str): Path where the output GIF will be saved.
        mapbox_token (str): Mapbox access token.
        overlap_threshold (float): Overlap threshold for clean_overlapping_bboxes. Defaults to 0.3.
        max_cleaning_iterations (int): Max number of cleaning passes (stops early if stable). Defaults to 50.
        frame_duration (int): Duration of each frame in milliseconds. Defaults to 1000.
        loop (int): Number of times the GIF loops (0 = infinite). Defaults to 0.
        image_cache_dir (str | Path | None): Directory to cache satellite images.
            If provided, images are saved/loaded from this directory to avoid re-fetching.
            Defaults to None (no caching).

    Returns:
        Path: Path to the saved GIF file.
    """
    lat, lon = coordinates
    img_width = 512
    img_height = 512 + 30  # 30px for Mapbox watermark

    # ---- 1. Create bounding box ------------------------------------------------
    bbox = create_bbox_from_coordinates(
        lat=lat,
        lon=lon,
        img_height=img_height,
        img_width=img_width,
    )

    # ---- 2. Fetch satellite image from Mapbox (with cache) ---------------------
    cache_path = None
    if image_cache_dir is not None:
        cache_dir = Path(image_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"satellite_{lat}_{lon}.png"

    if cache_path and cache_path.exists():
        image = Image.open(cache_path)
        print(f"Loaded cached image from {cache_path}")
    else:
        save_path = cache_path if cache_path else Path("satellite_tmp.png")
        image = ask_mapbox_for_image(
            output_file=save_path,
            image_width_height={"width": img_width, "height": img_height},
            mapbox_token=mapbox_token,
            bounding_box=bbox,
        )
        if cache_path:
            print(f"Cached image saved to {cache_path}")

    # ---- 3. Crop the 30px watermark at the bottom ------------------------------
    base_image = image.crop(box=(0, 0, img_width, img_height - 30))

    frames: list[Image.Image] = []

    # ---- 4. First frame : satellite background only ----------------------------
    frames.append(base_image.copy())

    # ---- 5. Fetch building polygons from OSM -----------------------------------
    west, south, east, north = bbox["west"], bbox["south"], bbox["east"], bbox["north"]
    gdf = ox.features_from_bbox(
        bbox=(west, south, east, north), tags={"building": True}
    )
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf = gdf.reset_index(drop=True)

    # Convert to bounding boxes in Lambert 93 (projected CRS for area operations)
    gdf_l93 = gdf.to_crs(epsg=2154)

    # Simplify gdf to geometry-only for clean operations
    gdf_bbox = gpd.GeoDataFrame(
        geometry=gdf_l93.geometry.apply(lambda g: box(*g.bounds)),
        crs=gdf_l93.crs,
    )

    # ---- 6 & 7. Draw bboxes, run one cleaning pass per frame --------------------
    for i in range(max_cleaning_iterations):
        n_bboxes = len(gdf_bbox)
        counter = f"Iter {i} | {n_bboxes} bboxes"

        # 6. Draw current state
        gdf_bbox_wgs = gdf_bbox.to_crs(epsg=4326)
        frames.append(
            _draw_geometries_on_image(
                base_image,
                gdf_bbox_wgs,
                bbox,
                outline_color="red",
                counter_text=counter,
            )
        )

        # 7. One full cleaning pass
        gdf_bbox = clean_overlapping_bboxes(
            gdf_bbox,
            threshold=overlap_threshold,
        )

    # Final frame
    n_bboxes = len(gdf_bbox)
    counter = f"Final | {n_bboxes} bboxes"
    gdf_bbox_wgs = gdf_bbox.to_crs(epsg=4326)
    frames.append(
        _draw_geometries_on_image(
            base_image,
            gdf_bbox_wgs,
            bbox,
            outline_color="red",
            counter_text=counter,
        )
    )

    # ---- Save GIF ---------------------------------------------------------------
    output_path = Path(output_file)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=loop,
    )

    return output_path


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()

    MAPBOX_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")
    if not MAPBOX_TOKEN:
        raise ValueError("MAPBOX_ACCESS_TOKEN not found in environment variables")

    output_gif = make_a_gif_for_coordinates(
        coordinates=(45.758683831005314, 4.835169445671655),  # Place in Lyon
        output_file="lyon_building_detection.gif",
        mapbox_token=MAPBOX_TOKEN,
        overlap_threshold=0.5,
        max_cleaning_iterations=5,
        frame_duration=1000,
        loop=0,
        image_cache_dir="data/scraped_images",
    )
    print(f"GIF saved to: {output_gif}")
