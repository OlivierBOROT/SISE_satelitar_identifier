import math
import os
from io import BytesIO

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from pyproj import Transformer
from shapely.geometry import box

load_dotenv()
# =====================
# PARAMÈTRES
# =====================
MAPBOX_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")
LAT = 45.86038477803
LON = 6.186636636094701

PIXEL_SIZE_M = 0.4  # 30–50 cm par pixel
IMG_SIZE = 512  # 512x512 px
OUTPUT_DIR = "test2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# 1️⃣ BBOX métrique centrée
# =====================
half_size_m = (IMG_SIZE * PIXEL_SIZE_M) / 2

# reprojection WGS84 -> Lambert 93
to_l93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
to_wgs84 = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

cx, cy = to_l93.transform(LON, LAT)

minx, miny = cx - half_size_m, cy - half_size_m
maxx, maxy = cx + half_size_m, cy + half_size_m

# retour en lat/lon pour Mapbox
west, south = to_wgs84.transform(minx, miny)
east, north = to_wgs84.transform(maxx, maxy)

print(f"BBOX (lat/lon) : {south}, {west}, {north}, {east}")
# =====================
# 2️⃣ Image satellite Mapbox
# =====================
img_url = (
    "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
    f"[{west},{south},{east},{north}]/{IMG_SIZE}x{IMG_SIZE}"
    f"?access_token={MAPBOX_TOKEN}"
)

img = requests.get(img_url).content
image = Image.open(BytesIO(img))
image.save(f"{OUTPUT_DIR}/image_satellite.png")

# =====================
# 3️⃣ Bâtiments OSM
# =====================
gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags={"building": True})

gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
gdf = gdf.to_crs(epsg=2154)

# sauvegarde polygones
gdf.to_crs(epsg=4326).to_file(
    f"{OUTPUT_DIR}/batiments_polygones.geojson", driver="GeoJSON"
)

# =====================
# 4️⃣ BBox minimale par bâtiment
# =====================
gdf_bbox = gdf.copy()
gdf_bbox["geometry"] = gdf_bbox.geometry.apply(lambda g: box(*g.bounds))

gdf_bbox.to_crs(epsg=4326).to_file(
    f"{OUTPUT_DIR}/batiments_bbox.geojson", driver="GeoJSON"
)

# =====================
# 5️⃣ Fusion image + polygones
# =====================
draw = ImageDraw.Draw(image)


def world_to_pixel(x, y):
    px = int((x - minx) / (maxx - minx) * IMG_SIZE)
    py = IMG_SIZE - int((y - miny) / (maxy - miny) * IMG_SIZE)
    return px, py


for geom in gdf.geometry:
    if geom.geom_type == "Polygon":
        coords = geom.exterior.coords
    else:
        coords = list(geom.geoms[0].exterior.coords)

    pixels = [world_to_pixel(x, y) for x, y in coords]
    draw.line(pixels, fill="red", width=2)

image.save(f"{OUTPUT_DIR}/fusion_image_polygones.png")

print("✅ Terminé")
print(f"📂 Résultats dans : {OUTPUT_DIR}/")
