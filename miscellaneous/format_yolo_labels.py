import geopandas as gpd

CLASS_IDS = {
    'house': 1,
    'apartments': 2
}


def format_yolo_labels(gdf: gpd.GeoDataFrame, bbox: tuple) -> list[str]:
    """
    Convert a OSM features to YOLO format labels:
    ```txt
    class_id x_center y_center width height
    ```

    Args:
        gdf (gpd.GeoDataFrame): OpenStreetMap features
        bbox (tuple): Bounding box used for image

    Returns:
        str: YOLO formated labels
    """
    img_minx, img_miny, img_maxx, img_maxy = bbox
    img_w = img_maxx - img_minx
    img_h = img_maxy - img_miny

    bounds = gdf.bounds
    gdf = gdf.join(bounds)

    gdf['x_center'] = ((gdf['minx'] + gdf['maxx']) / 2 - img_minx) / img_w
    gdf['y_center'] = (img_maxy - (gdf['miny'] + gdf['maxy']) / 2) / img_h  # note the flip
    gdf['width'] = (gdf['maxx'] - gdf['minx']) / img_w
    gdf['height'] = (gdf['maxy'] - gdf['miny']) / img_h

    gdf['class_id'] = gdf['building'].map(CLASS_IDS).fillna(0)

    yolo_lines = (
        gdf[['class_id', 'x_center', 'y_center', 'width', 'height']]
        .apply(lambda r: f"{int(r.class_id)} {r.x_center:.6f} {r.y_center:.6f} "
                         f"{r.width:.6f} {r.height:.6f}", axis=1)
        .tolist()
    )

    return yolo_lines

