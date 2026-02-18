import os

import geopandas as gpd
import osmnx as ox
import pandas as pd
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon, box

# =========================
# PARAMÈTRES À MODIFIER
# =========================
LAT = 48.28373534674075
LON = -2.9453320499772366
SEARCH_RADIUS_M = 120  # rayon autour du point en mètres
OVERLAP_THRESHOLD = 0.05

BASE_DIR = "testing"
RAW_DIR = os.path.join(BASE_DIR, "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "clean")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

# =========================
# 1️⃣ Conversion coordonnées → bbox lat/lon
# =========================
to_l93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
to_wgs = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

cx, cy = to_l93.transform(LON, LAT)
minx, miny = cx - SEARCH_RADIUS_M, cy - SEARCH_RADIUS_M
maxx, maxy = cx + SEARCH_RADIUS_M, cy + SEARCH_RADIUS_M

west, south = to_wgs.transform(minx, miny)
east, north = to_wgs.transform(maxx, maxy)

# =========================
# 2️⃣ Récupération bâtiments OSM
# =========================
gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags={"building": True})

gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
gdf = gdf.reset_index(drop=True)

# =========================
# 3️⃣ Export RAW
# =========================
gdf.to_file(os.path.join(RAW_DIR, "batiments_polygones_raw.geojson"), driver="GeoJSON")

# BBox brutes RAW
gdf_bbox_raw = gdf.to_crs(epsg=2154).copy()
gdf_bbox_raw["geometry"] = gdf_bbox_raw.geometry.apply(lambda g: box(*g.bounds))
gdf_bbox_raw.to_crs(epsg=4326).to_file(
    os.path.join(RAW_DIR, "batiments_bbox_raw.geojson"), driver="GeoJSON"
)


# =========================
# 4️⃣ Nettoyage des bbox (overlap > 30%)
# =========================
def clean_overlapping_bboxes(
    gdf, threshold=0.3, what_to_do_on_overlap="remove_smaller"
):
    """
    Supprime la plus petite bbox si overlap > threshold.
    Les plus grandes sont traitées en premier.
    """

    for polygon in gdf.geometry:
        if not isinstance(polygon, (Polygon, MultiPolygon)):
            raise ValueError("La géométrie doit être un Polygon ou MultiPolygon")
        for other in gdf.geometry:
            if polygon == other:
                continue
            inter = polygon.intersection(other).area
            union = polygon.union(other).area
            iou = inter / union
            if iou > threshold:
                if what_to_do_on_overlap == "remove_smaller":
                    # supprimer la plus petite bbox
                    if polygon.area >= other.area:
                        gdf = gdf[gdf.geometry != other]
                    else:
                        gdf = gdf[gdf.geometry != polygon]
                elif what_to_do_on_overlap == "merge":
                    # calculer la nouvelle bbox
                    new_polygon = polygon.union(other).envelope
                    # supprimer les deux anciennes et ajouter la nouvelle
                    gdf = gdf[gdf.geometry != polygon]
                    gdf = gdf[gdf.geometry != other]
                    gdf = pd.concat(
                        [gdf, gpd.GeoDataFrame([{"geometry": new_polygon}])],
                        ignore_index=True,
                    )

    return gdf


gdf_bbox_clean = clean_overlapping_bboxes(gdf_bbox_raw, OVERLAP_THRESHOLD, "merge")
gdf_bbox_clean = clean_overlapping_bboxes(gdf_bbox_clean, OVERLAP_THRESHOLD, "merge")
gdf_bbox_clean.to_crs(epsg=4326).to_file(
    os.path.join(CLEAN_DIR, "batiments_bbox_clean.geojson"), driver="GeoJSON"
)

# =========================
# 5️⃣ Résumé
# =========================
print("✅ Traitement terminé")
print(f"📍 Point : {LAT}, {LON}")
print(f"RAW  bbox : {len(gdf_bbox_raw)}")
print(f"CLEAN bbox : {len(gdf_bbox_clean)}")
print(f"📂 Résultats dans {RAW_DIR} et {CLEAN_DIR}")

# continuer jusqu'à changement du nombre de polygones
