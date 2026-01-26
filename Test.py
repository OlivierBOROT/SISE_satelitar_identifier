import geojson
import requests
from shapely.geometry import MultiPolygon, Polygon

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def fetch_buildings(bbox):
    """
    bbox = (south, west, north, east)
    """
    query = f"""
    [out:json];
    (
      way["building"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
      relation["building"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    out geom;
    """

    response = requests.post(OVERPASS_URL, data=query)
    response.raise_for_status()
    return response.json()


def osm_to_geojson(osm_data):
    features = []

    for el in osm_data["elements"]:
        if "geometry" not in el:
            continue

        coords = [(p["lon"], p["lat"]) for p in el["geometry"]]

        # fermer le polygone si nécessaire
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        polygon = Polygon(coords)

        if not polygon.is_valid:
            continue

        feature = geojson.Feature(
            geometry=polygon,
            properties={
                "osm_id": el["id"],
                "type": el.get("tags", {}).get("building", "yes"),
            },
        )
        features.append(feature)

    return geojson.FeatureCollection(features)


if __name__ == "__main__":
    # Exemple : Paris (quartier)
    bbox = (48.8563, 2.3499, 48.8573, 2.3510)

    osm_data = fetch_buildings(bbox)
    geojson_data = osm_to_geojson(osm_data)

    with open("batiments.geojson", "w", encoding="utf-8") as f:
        geojson.dump(geojson_data, f, ensure_ascii=False)

    print(f"✅ {len(geojson_data.features)} bâtiments exportés")
