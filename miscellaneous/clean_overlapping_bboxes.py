"""Provides functionality to clean overlapping bounding boxes in a GeoDataFrame."""

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon


def clean_overlapping_bboxes(
    gdf: gpd.GeoDataFrame,
    threshold: float = 0.3,
) -> gpd.GeoDataFrame:
    """Perform a single pass of overlap-based merging on a GeoDataFrame of polygons.

    For each polygon, check if it intersects with another polygon. If the
    intersection area represents more than *threshold* percent of the polygon's
    own area, the two shapes are merged (union + envelope).

    Only **one pass** is performed: the function scans all polygons once and
    merges the first overlapping pair it finds for each polygon.  Call this
    function repeatedly until the result stabilises to fully clean the data.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing the bounding boxes.
        threshold (float, optional): If intersection_area / polygon_area
            exceeds this value the pair is merged. Defaults to 0.3.

    Raises:
        ValueError: If any geometry is not a Polygon or MultiPolygon.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame after one merging pass.
    """
    gdf = gdf.copy().reset_index(drop=True)

    for geom in gdf.geometry:
        if not isinstance(geom, (Polygon, MultiPolygon)):
            raise ValueError("Geometry must be a Polygon or MultiPolygon")

    merged_indices: set[int] = set()
    new_geometries: list[Polygon] = []

    for idx_a in gdf.index:
        if idx_a in merged_indices:
            continue

        geom_a = gdf.loc[idx_a, "geometry"]
        if geom_a.area == 0:
            merged_indices.add(idx_a)
            continue

        merge_target: int | None = None

        for idx_b in gdf.index:
            if idx_b <= idx_a or idx_b in merged_indices:
                continue

            geom_b = gdf.loc[idx_b, "geometry"]
            if geom_b.area == 0:
                continue

            inter_area = geom_a.intersection(geom_b).area

            # Check if either polygon is overlapped above threshold
            ratio_a = inter_area / geom_a.area
            ratio_b = inter_area / geom_b.area

            if ratio_a > threshold or ratio_b > threshold:
                merge_target = idx_b
                break

        if merge_target is not None:
            geom_b = gdf.loc[merge_target, "geometry"]
            new_geometries.append(geom_a.union(geom_b).envelope)
            merged_indices.add(idx_a)
            merged_indices.add(merge_target)
        # else: polygon stays as-is (handled below)

    # Build result: keep un-merged polygons + add newly merged ones
    remaining = gdf.loc[~gdf.index.isin(merged_indices)]
    if new_geometries:
        merged_gdf = gpd.GeoDataFrame(
            [{"geometry": g} for g in new_geometries], crs=gdf.crs
        )
        result = gpd.GeoDataFrame(
            pd.concat([remaining, merged_gdf], ignore_index=True),
            geometry="geometry",
            crs=gdf.crs,
        )
    else:
        result = remaining.reset_index(drop=True)

    return result
