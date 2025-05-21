# processors/geometry_utils.py
from shapely.geometry import Polygon

def get_bounding_box(polygon):
    """Get the bounding box of a polygon"""
    minx, miny, maxx, maxy = polygon.bounds
    return Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])

def get_tile_centroid(tile_polygon):
    """Get the centroid of a tile polygon"""
    return tile_polygon.centroid