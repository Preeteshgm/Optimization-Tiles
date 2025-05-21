# processors/VisualizationProcessor.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import random
from shapely.geometry import Polygon
import io
import base64

class VisualizationProcessor:
    def __init__(self):
        self.colors = {}
        self.apartment_names = {}

    def plot_room_boundaries(self, rooms, start_points=None):
        plt.figure(figsize=(12, 12))
        for room in rooms:
            x, y = room.exterior.xy
            plt.plot(x, y, color='blue', linewidth=1.5)
        if start_points:
            for sp in start_points:
                cx, cy = sp['centroid']
                plt.plot(cx, cy, 'ro', markersize=8)
                plt.text(cx, cy, 'SP', fontsize=14, ha='center', color='red')
        plt.title("Room Boundaries with Tile Start Points")
        plt.grid(True)
        
        # Return the figure for web display rather than showing it
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def plot_clusters(self, clusters_df, use_final_names=False):
        plt.figure(figsize=(14, 14))
        unique_clusters = clusters_df['apartment_cluster'].unique()

        # Use final names if available, otherwise default cluster names
        if use_final_names and 'apartment_name' in clusters_df.columns:
            apartment_names = clusters_df.groupby('apartment_cluster')['apartment_name'].first().to_dict()
            self.apartment_names.update(apartment_names)
        else:
            apartment_names = {cluster_id: f"Apartment {cluster_id+1}" for cluster_id in unique_clusters}

        for cluster_id in unique_clusters:
            cluster_rooms = clusters_df[clusters_df['apartment_cluster'] == cluster_id]
            color = self.get_color(cluster_id)
            apt_name = self.apartment_names.get(cluster_id, apartment_names.get(cluster_id, f"Apartment {cluster_id+1}"))
            for idx, row in cluster_rooms.iterrows():
                polygon = row['polygon']
                x, y = polygon.exterior.xy
                plt.fill(x, y, alpha=0.6, label=apt_name if idx == cluster_rooms.index[0] else "", color=color)
                plt.text(row['centroid_x'], row['centroid_y'], row['room_name'], fontsize=14, ha='center', color='black')
        plt.title("Apartment Clusters with Default or Final Room Names")
        plt.legend()
        plt.grid(True)
        
        # Return the figure for web display rather than showing it
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def get_color(self, cluster_id):
        if cluster_id not in self.colors:
            self.colors[cluster_id] = (random.random(), random.random(), random.random())
        return self.colors[cluster_id]