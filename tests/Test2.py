import osmnx as ox

north, south, east, west = 48.8573, 48.8563, 2.3510, 2.3499

gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags={"building": True})

# garder uniquement les géométries surfaciques
gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

gdf.to_file("batiments2.geojson", driver="GeoJSON")

print(f"✅ {len(gdf)} bâtiments exportés")
