import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import box

# bbox de la zone
south, west, north, east = (
    45.85950061935637,
    6.185264716402588,
    45.861268920067225,
    6.188008599760147,
)

# récupération des bâtiments
gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags={"building": True})

# garder uniquement les polygones
gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

# ---- 1️⃣ Polygones originaux ----
gdf_poly = gdf.copy()
gdf_poly["geom_type"] = "building_polygon"

# ---- 2️⃣ Bounding boxes ----
gdf_bbox = gdf.copy()
gdf_bbox["geometry"] = gdf_bbox.geometry.apply(lambda g: box(*g.bounds))
gdf_bbox["geom_type"] = "building_bbox"

# ---- 3️⃣ Fusion des deux ----
gdf_all = gpd.GeoDataFrame(pd.concat([gdf_bbox], ignore_index=True), crs=gdf.crs)

# ---- 4️⃣ Export ----
gdf_all.to_file("batiments_polygones_et_bbox2.geojson", driver="GeoJSON")

print(f"✅ {len(gdf)} bâtiments → {len(gdf_all)} features exportées")
