# processors/RoomClusterProcessor.py

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon

class RoomClusterProcessor:
    def preview_clusters(self):
        if self.room_df is None or self.room_df.empty:
            print("❌ No rooms clustered yet. Please run cluster_rooms() first.")
            return

        print("🗺️ Previewing clusters with default names:")
        for apt_name, room_names in self.apartment_names.items():
            print(f"🏢 {apt_name}:")
            for room_name in room_names:
                print(f"   - {room_name}")
        print(f"✅ Preview complete. {len(self.apartment_names)} apartments found.")

    def __init__(self, eps=5000, min_samples=1):
        self.eps = eps
        self.min_samples = min_samples
        self.room_df = None
        self.apartment_names = {}

    def cluster_rooms(self, rooms):
        print("🔍 Clustering rooms into apartments...")
        room_centroids = np.array([[room.centroid.x, room.centroid.y] for room in rooms])
        self.room_df = pd.DataFrame({
            'room_id': range(len(rooms)),
            'centroid_x': room_centroids[:, 0],
            'centroid_y': room_centroids[:, 1],
            'polygon': rooms
        })
        self.room_df['apartment_cluster'] = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(room_centroids)
        print(f"✅ Clustered into {self.room_df['apartment_cluster'].nunique()} apartments.")
        return self.room_df

    def assign_default_names(self):
        print("🔤 Assigning default apartment and room names...")
        self.apartment_names = {}
        for apt_id in sorted(self.room_df['apartment_cluster'].unique()):
            apt_rooms = self.room_df[self.room_df['apartment_cluster'] == apt_id]
            apt_name = f"A{apt_id+1}"
            room_names = [f"{apt_name}-R{idx+1}" for idx in range(len(apt_rooms))]
            self.apartment_names[apt_name] = room_names
            self.room_df.loc[self.room_df['apartment_cluster'] == apt_id, 'room_name'] = room_names
        print(f"✅ Default names assigned: {self.apartment_names}")
        return self.apartment_names

    def prompt_for_names(self):
        print("📝 Assigning final apartment and room names...")
        for apt_id in sorted(self.room_df['apartment_cluster'].unique()):
            apt_rooms = self.room_df[self.room_df['apartment_cluster'] == apt_id]
            apt_name = input(f"Enter a name for Apartment {apt_id+1} (e.g., A1, B1): ").strip()
            room_names = []
            for idx, row in apt_rooms.iterrows():
                room_name = input(f"Enter a name for Room {row['room_id']+1} in {apt_name} (e.g., Living Room, Bedroom 1): ").strip()
                room_names.append(room_name)
            self.apartment_names[apt_name] = room_names
            self.room_df.loc[self.room_df['apartment_cluster'] == apt_id, 'room_name'] = room_names
        print(f"✅ Final names assigned: {self.apartment_names}")
        return self.apartment_names

    def get_room_dataframe(self):
        return self.room_df

    def get_apartment_names(self):
        return self.apartment_names