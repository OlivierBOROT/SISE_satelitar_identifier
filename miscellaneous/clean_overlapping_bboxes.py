"""Provides functionality to clean overlapping bounding boxes in a GeoDataFrame."""
import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

def clean_overlapping_bboxes(
    gdf: gpd.GeoDataFrame,
    threshold: float = 0.3,
    what_to_do_on_overlap: str = "remove_smaller"
) -> gpd.GeoDataFrame:
    """
    Provides functionality to clean overlapping bounding boxes in a GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing the bounding boxes.
        threshold (float, optional): Overlap threshold to consider. Defaults to 0.3.
        what_to_do_on_overlap (str, optional): Action to take on overlap.
        Defaults to "remove_smaller".
        Options are "remove_smaller" or "merge".

    Raises:
        ValueError: If any geometry in the GeoDataFrame is not a Polygon or MultiPolygon.

    Returns:
        gpd.GeoDataFrame: Cleaned GeoDataFrame with overlapping bounding boxes handled as specified.
    """

    for polygon in gdf.geometry:
        if not isinstance(polygon, (Polygon, MultiPolygon)):
            raise ValueError("Geometry must be a Polygon or MultiPolygon")
        for other in gdf.geometry:
            if polygon == other:
                continue
            inter = polygon.intersection(other).area
            union = polygon.union(other).area
            iou = inter / union
            if iou > threshold:
                if what_to_do_on_overlap == "remove_smaller":
                    # remove the smaller bbox
                    if polygon.area >= other.area:
                        gdf = gdf[gdf.geometry != other]
                    else:
                        gdf = gdf[gdf.geometry != polygon]
                elif what_to_do_on_overlap == "merge":
                    # calculate the new bbox
                    new_polygon = polygon.union(other).envelope
                    # remove the two old ones and add the new one
                    gdf = gdf[gdf.geometry != polygon]
                    gdf = gdf[gdf.geometry != other]
                    gdf = gpd.GeoDataFrame(
                        pd.concat(
                            [gdf, gpd.GeoDataFrame([{"geometry": new_polygon}], crs=gdf.crs)],
                            ignore_index=True,
                        ),
                        geometry="geometry",
                        crs=gdf.crs,
                    )

    return gdf
