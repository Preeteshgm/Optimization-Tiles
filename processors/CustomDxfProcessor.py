# processors/CustomDxfProcessor.py

import ezdxf
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPolygon, LineString
from shapely.ops import unary_union
from collections import defaultdict
from sklearn.cluster import KMeans
import pandas as pd
import math
import os

class CustomDxfProcessor:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.doc = None
        self.room_polygons = []
        self.start_points = []
        self.sp_polygons = []
        self.tiles = []

    def load_dxf(self):
        try:
            # Load the DXF file
            self.doc = ezdxf.readfile(self.file_path)
            print("✅ DXF file loaded successfully.")
            return True
        except Exception as e:
            print(f"❌ Error loading DXF file: {e}")
            return False

    def list_all_layers(self):
        """List all layers in the DXF file"""
        if not self.doc:
            if not self.load_dxf():
                return []
        layers = set()
        for entity in self.doc.modelspace():
            if hasattr(entity.dxf, 'layer'):
                layers.add(entity.dxf.layer)
        return sorted(list(layers))

    def collect_and_clean_linework(self, layer_name, tol=1e-2):
        raw_lines = []
        for entity in self.doc.modelspace():
            if hasattr(entity.dxf, 'layer') and entity.dxf.layer.lower() == layer_name.lower():
                if entity.dxftype() == 'LINE':
                    a = (entity.dxf.start.x, entity.dxf.start.y)
                    b = (entity.dxf.end.x, entity.dxf.end.y)
                    raw_lines.append((a, b))
                elif entity.dxftype() == 'LWPOLYLINE':
                    pts = [(pt[0], pt[1]) for pt in entity.get_points()]
                    if entity.closed:
                        pts.append(pts[0])
                    for i in range(len(pts) - 1):
                        raw_lines.append((pts[i], pts[i+1]))
                elif entity.dxftype() == 'POLYLINE':
                    pts = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
                    if pts[0] != pts[-1]:  # Ensure closed loop
                        pts.append(pts[0])
                    for i in range(len(pts) - 1):
                        raw_lines.append((pts[i], pts[i+1]))
        return self.clean_and_snap_lines(raw_lines, tol)

    def clean_and_snap_lines(self, raw_lines, tol=1e-2):
        all_pts = list(set([pt for line in raw_lines for pt in line]))
        snapped_pts = []

        def find_or_add(pt):
            for s in snapped_pts:
                if np.linalg.norm(np.array(pt) - np.array(s)) < tol:
                    return s
            snapped_pts.append(pt)
            return pt

        clean_lines = []
        seen = set()
        for a, b in raw_lines:
            a_snap = find_or_add(a)
            b_snap = find_or_add(b)
            if a_snap == b_snap:
                continue
            key = tuple(sorted([a_snap, b_snap]))
            if key in seen:
                continue
            seen.add(key)
            clean_lines.append((a_snap, b_snap))

        return clean_lines

    def build_point_graph(self, lines):
        graph = defaultdict(set)
        for a, b in lines:
            graph[a].add(b)
            graph[b].add(a)
        return graph

    def find_closed_loops(self, graph, max_depth=30):
        visited_edges = set()
        loops = []

        def dfs(path, start):
            current = path[-1]
            for neighbor in graph[current]:
                edge = tuple(sorted((current, neighbor)))
                if edge in visited_edges:
                    continue
                if neighbor == start and len(path) >= 3:
                    loop = path + [neighbor]
                    loops.append(loop)
                    for i in range(len(loop) - 1):
                        visited_edges.add(tuple(sorted((loop[i], loop[i + 1]))))
                    return
                elif neighbor not in path and len(path) < max_depth:
                    dfs(path + [neighbor], start)

        for pt in graph:
            dfs([pt], pt)

        return loops

    def heal_graph_gaps(self, lines, snap_tol=5.0):
        graph = self.build_point_graph(lines)
        endpoints = [pt for pt, nbrs in graph.items() if len(nbrs) == 1]
        healed_lines = list(lines)
        added = 0

        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                a = endpoints[i]
                b = endpoints[j]
                dist = np.linalg.norm(np.array(a) - np.array(b))
                if dist < snap_tol:
                    healed_lines.append((a, b))
                    added += 1

        print(f"🔗 Healed {added} gaps (connected dangling points)")
        return healed_lines

    def extract_room_boundaries(self, layer_name="Tile Layout"):
        print("Extracting room boundaries...")
        lines = self.collect_and_clean_linework(layer_name)
        if not lines:
            print(f"No lines found in layer '{layer_name}'")
            return []

        # Heal gaps in the line network
        lines = self.heal_graph_gaps(lines, snap_tol=5.0)

        # Build graph and find closed loops
        graph = self.build_point_graph(lines)
        loops = self.find_closed_loops(graph)

        # Convert loops to polygons
        polygons = []
        for loop in loops:
            poly = Polygon(loop)
            if poly.is_valid and poly.area > 0:
                polygons.append(poly)

        self.room_polygons = polygons
        print(f"✅ Found {len(polygons)} room boundary polygons.")
        return polygons

    def extract_start_points(self, layer_name="SP"):
        print("Extracting starting points from 'SP' layer...")
        start_points = []
        for entity in self.doc.modelspace():
            if hasattr(entity.dxf, 'layer') and entity.dxf.layer == layer_name:
                if entity.dxftype() == 'LWPOLYLINE' and entity.closed:
                    vertices = [(pt[0], pt[1]) for pt in entity.get_points()]
                    poly = Polygon(vertices)
                    if poly.is_valid and poly.area > 0:
                        centroid = poly.centroid
                        start_points.append({
                            'polygon': poly,
                            'centroid': (centroid.x, centroid.y),
                            'width': round(poly.bounds[2] - poly.bounds[0], 2),
                            'height': round(poly.bounds[3] - poly.bounds[1], 2),
                            'area': poly.area
                        })

        self.start_points = start_points
        print(f"✅ Found {len(start_points)} starting points.")
        return start_points

    def extract_tile_sizes_from_sp(self, layer_name="SP"):
        print("Extracting tile sizes from SP layer...")
        self.sp_polygons = self.extract_start_points(layer_name=layer_name)
        tile_sizes = {(sp['width'], sp['height']) for sp in self.sp_polygons}
        print(f"✅ Extracted {len(tile_sizes)} unique tile sizes from SP layer.")
        return list(tile_sizes)