# processors/TileProcessor.py

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.affinity import rotate, translate

# Import from geometry_utils
from processors.geometry_utils import get_bounding_box, get_tile_centroid

class TileProcessor:
    def __init__(self, dxf_processor=None, tile_width=600, tile_height=600, grout_thickness=3):
        self.dxf_processor = dxf_processor
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.grout_thickness = grout_thickness
        self.tiles = []
        self.start_points = []

    def set_tile_size(self, width, height):
        self.tile_width = width
        self.tile_height = height
        print(f"✅ Tile size set to {width} x {height} mm")

    def set_grout_thickness(self, thickness):
        self.grout_thickness = thickness
        print(f"✅ Grout thickness set to {thickness} mm")

    def set_start_points(self, start_points):
        self.start_points = start_points
        print(f"✅ Set {len(start_points)} start points for tile placement")

    def generate_tile_grid(self, room_poly, orientation=0):
        """Basic tile grid generation (no grout) - kept for backward compatibility"""
        print(f"🧩 Generating tiles for room with orientation {orientation}°...")
        room_bounds = get_bounding_box(room_poly)
        tile_w, tile_h = self.tile_width, self.tile_height

        # Apply orientation
        if orientation == 90:
            tile_w, tile_h = tile_h, tile_w

        # Calculate the grid of tiles
        min_x, min_y, max_x, max_y = room_bounds.bounds
        tiles = []

        x = min_x
        while x < max_x:
            y = min_y
            while y < max_y:
                tile = Polygon([
                    (x, y),
                    (x + tile_w, y),
                    (x + tile_w, y + tile_h),
                    (x, y + tile_h),
                    (x, y)
                ])

                # Rotate and translate to fit the room orientation
                tile = rotate(tile, orientation, origin='center', use_radians=False)

                # Check if the tile is inside the room
                if room_poly.intersects(tile):
                    centroid = get_tile_centroid(tile)
                    # Calculate the percentage of the tile that intersects with the room
                    intersection = room_poly.intersection(tile)
                    intersection_percentage = intersection.area / tile.area

                    # Only include tiles that have at least 50% of their area inside the room
                    if intersection_percentage >= 0.5:
                        tiles.append({
                            'polygon': tile,
                            'centroid_x': centroid.x,
                            'centroid_y': centroid.y,
                            'width': tile_w,
                            'height': tile_h,
                            'orientation': orientation,
                            'area': tile.area,
                            'intersection_percentage': intersection_percentage
                        })
                y += tile_h
            x += tile_w

        print(f"✅ Generated {len(tiles)} tiles for the room")
        return tiles

    def generate_aligned_tile_grid(self, room_poly, orientation=0, start_point=None, 
                                 stagger_percent=0, stagger_direction='x', room_id=-1,
                                 sp_includes_grout=True):
        """
        Generate grid-aligned tiles with explicit grout spacing
        
        Parameters:
        -----------
        room_poly : Polygon
            The room boundary polygon
        orientation : int
            Tile orientation in degrees (0 or 90)
        start_point : Point or tuple, optional
            Starting point for tile layout
        stagger_percent : float
            Stagger percentage (0.0 to 1.0) for brick pattern
        stagger_direction : str
            Direction of stagger ('x' or 'y')
        room_id : int
            Room identifier
        sp_includes_grout : bool
            Whether the specified tile size includes grout
            
        Returns:
        --------
        List of dict with tile information
        """
        print(f"🧭 Generating aligned tiles with grout for room {room_id} with orientation {orientation}°...")
        
        # Process tile dimensions based on whether SP includes grout
        if sp_includes_grout:
            layout_tw, layout_th = self.tile_width, self.tile_height  # Layout size (with grout)
            actual_tile_width = self.tile_width - self.grout_thickness  # Factory size
            actual_tile_height = self.tile_height - self.grout_thickness  # Factory size
            print(f"   SP layer tile size (with grout): {layout_tw}x{layout_th} mm")
            print(f"   Factory tile size (without grout): {actual_tile_width}x{actual_tile_height} mm")
        else:
            actual_tile_width, actual_tile_height = self.tile_width, self.tile_height  # Factory size
            layout_tw = self.tile_width + self.grout_thickness  # Layout size (with grout)
            layout_th = self.tile_height + self.grout_thickness  # Layout size (with grout)
            print(f"   SP layer tile size (factory size): {actual_tile_width}x{actual_tile_height} mm")
            print(f"   Layout tile size (with grout): {layout_tw}x{layout_th} mm")

        # Convert start_point to proper format
        if start_point is None:
            start_point = room_poly.centroid
            print(f"   Using room centroid as start point: ({start_point.x}, {start_point.y})")
        elif isinstance(start_point, Point):
            print(f"   Using provided start point: ({start_point.x}, {start_point.y})")
        else:
            start_point = Point(start_point[0], start_point[1])
            print(f"   Using provided start point: ({start_point.x}, {start_point.y})")

        # Apply orientation to layout and actual tile sizes
        if orientation == 90:
            layout_tw, layout_th = layout_th, layout_tw
            actual_tile_width, actual_tile_height = actual_tile_height, actual_tile_width
            # Flip stagger direction if orientation is 90°
            stagger_direction = 'y' if stagger_direction == 'x' else 'x'
            print(f"   Applied 90° rotation: layout size {layout_tw}x{layout_th}, tile size {actual_tile_width}x{actual_tile_height}")

        # Calculate stagger size
        stagger_size = layout_tw * stagger_percent if stagger_direction == 'x' else layout_th * stagger_percent
        if stagger_percent > 0:
            print(f"   Using stagger: {stagger_percent*100}% in {stagger_direction} direction")

        room_tiles = []

        # Get room bounds with extension to ensure complete coverage
        minx, miny, maxx, maxy = room_poly.bounds
        extension = max(layout_tw, layout_th) * 0.5
        ext_minx, ext_miny = minx - extension, miny - extension
        ext_maxx, ext_maxy = maxx + extension, maxy + extension

        # Calculate optimal starting positions
        x_start = ext_minx - (ext_minx % layout_tw) if ext_minx % layout_tw != 0 else ext_minx
        y_start = ext_miny - (ext_miny % layout_th) if ext_miny % layout_th != 0 else ext_miny

        # Generate tiles with stagger logic and explicit grout spacing
        if stagger_direction == 'x':
            # Horizontal staggering
            y, row_counter = y_start, 0
            while y <= ext_maxy:
                x_offset = stagger_size if (row_counter % 2 == 1 and stagger_percent > 0) else 0
                x = x_start + x_offset

                while x <= ext_maxx:
                    # Center of the layout cell (includes grout space)
                    cx, cy = x + layout_tw / 2, y + layout_th / 2

                    # Create the layout tile polygon (with grout) for coverage calculations
                    layout_tile = Polygon([
                        (cx - layout_tw / 2, cy - layout_th / 2),
                        (cx + layout_tw / 2, cy - layout_th / 2),
                        (cx + layout_tw / 2, cy + layout_th / 2),
                        (cx - layout_tw / 2, cy + layout_th / 2)
                    ])

                    # Create the actual tile polygon (factory size without grout)
                    actual_tile = Polygon([
                        (cx - actual_tile_width / 2, cy - actual_tile_height / 2),
                        (cx + actual_tile_width / 2, cy - actual_tile_height / 2),
                        (cx + actual_tile_width / 2, cy + actual_tile_height / 2),
                        (cx - actual_tile_width / 2, cy + actual_tile_height / 2)
                    ])

                    # Process the tile if it intersects with the room
                    if layout_tile.is_valid and layout_tile.intersects(room_poly):
                        layout_intersection = layout_tile.intersection(room_poly)
                        if not layout_intersection.is_empty and layout_intersection.area > 0:
                            if actual_tile.intersects(room_poly):
                                actual_intersection = actual_tile.intersection(room_poly)
                                is_full = actual_intersection.equals(actual_tile)
                                
                                room_tiles.append({
                                    'polygon': actual_intersection,  # The actual visible tile
                                    'layout_polygon': layout_intersection,  # Layout cell (for coverage)
                                    'room_id': room_id,
                                    'width': layout_tw,  # Layout width (with grout)
                                    'height': layout_th,  # Layout height (with grout)
                                    'actual_tile_width': actual_tile_width,  # Factory width
                                    'actual_tile_height': actual_tile_height,  # Factory height
                                    'centroid': (cx, cy),
                                    'area': actual_intersection.area,
                                    'type': 'full' if is_full else 'cut',
                                    'orientation': orientation,
                                    'grout_thickness': self.grout_thickness
                                })
                    x += layout_tw
                y += layout_th
                row_counter += 1
        else:  # stagger_direction == 'y'
            # Vertical staggering
            x, column_counter = x_start, 0
            while x <= ext_maxx:
                y_offset = stagger_size if (column_counter % 2 == 1 and stagger_percent > 0) else 0
                y = y_start + y_offset

                while y <= ext_maxy:
                    cx, cy = x + layout_tw / 2, y + layout_th / 2
                    
                    layout_tile = Polygon([
                        (cx - layout_tw / 2, cy - layout_th / 2),
                        (cx + layout_tw / 2, cy - layout_th / 2),
                        (cx + layout_tw / 2, cy + layout_th / 2),
                        (cx - layout_tw / 2, cy + layout_th / 2)
                    ])

                    actual_tile = Polygon([
                        (cx - actual_tile_width / 2, cy - actual_tile_height / 2),
                        (cx + actual_tile_width / 2, cy - actual_tile_height / 2),
                        (cx + actual_tile_width / 2, cy + actual_tile_height / 2),
                        (cx - actual_tile_width / 2, cy + actual_tile_height / 2)
                    ])

                    if layout_tile.is_valid and layout_tile.intersects(room_poly):
                        layout_intersection = layout_tile.intersection(room_poly)
                        if not layout_intersection.is_empty and layout_intersection.area > 0:
                            if actual_tile.intersects(room_poly):
                                actual_intersection = actual_tile.intersection(room_poly)
                                is_full = actual_intersection.equals(actual_tile)
                                
                                room_tiles.append({
                                    'polygon': actual_intersection,
                                    'layout_polygon': layout_intersection,
                                    'room_id': room_id,
                                    'width': layout_tw,
                                    'height': layout_th,
                                    'actual_tile_width': actual_tile_width,
                                    'actual_tile_height': actual_tile_height,
                                    'centroid': (cx, cy),
                                    'area': actual_intersection.area,
                                    'type': 'full' if is_full else 'cut',
                                    'orientation': orientation,
                                    'grout_thickness': self.grout_thickness
                                })
                    y += layout_th
                x += layout_tw
                column_counter += 1

        # Verify coverage
        room_area = room_poly.area
        combined_tiles = None
        for tile in room_tiles:
            poly = tile['layout_polygon']
            if combined_tiles is None:
                combined_tiles = poly
            else:
                try:
                    combined_tiles = combined_tiles.union(poly)
                except:
                    try:
                        combined_tiles = combined_tiles.buffer(0).union(poly.buffer(0))
                    except:
                        continue

        if combined_tiles:
            coverage_pct = (combined_tiles.area / room_area) * 100
            if coverage_pct < 99.5:
                print(f"⚠️ Warning: Tiles cover only {coverage_pct:.2f}% of the room area")
            else:
                print(f"✅ Good coverage: Tiles cover {coverage_pct:.2f}% of the room area")

        full_count = len([t for t in room_tiles if t['type'] == 'full'])
        cut_count = len([t for t in room_tiles if t['type'] == 'cut'])
        print(f"✅ Generated {len(room_tiles)} tiles ({full_count} full, {cut_count} cut)")
        
        return room_tiles
        
    def generate_tiles_for_all_rooms(self, room_df, apartment_orientations, start_points=None, 
                                   stagger_percent=0, stagger_direction='x',
                                   sp_includes_grout=True):
        """Process all rooms to generate tiles with explicit grout spacing"""
        print("🔄 Processing tiles for all rooms with explicit grout spacing...")

        apartments_data = {}

        for _, room in room_df.iterrows():
            apartment_name = room['apartment_name']
            room_name = room['room_name']
            room_id = room['room_id']
            room_poly = room['polygon']

            # Get orientation for this apartment
            orientation = 0  # Default orientation
            for _, orient_row in apartment_orientations.iterrows():
                if orient_row['apartment_name'] == apartment_name:
                    orientation = orient_row['orientation']
                    break

            # Find start point and tile size for this room
            start_point = None
            room_tile_width = self.tile_width
            room_tile_height = self.tile_height
            
            if start_points:
                for sp in start_points:
                    if 'centroid' in sp and room_poly.contains(Point(sp['centroid'])):
                        start_point = Point(sp['centroid'])
                        if 'width' in sp and 'height' in sp:
                            room_tile_width = sp['width']
                            room_tile_height = sp['height']
                        break

            print(f"\nProcessing {apartment_name}-{room_name} (Room ID: {room_id}) with orientation {orientation}°...")
            
            # Use the room-specific tile size
            self.set_tile_size(room_tile_width, room_tile_height)
            
            # Generate tiles for this room
            tiles = self.generate_aligned_tile_grid(
                room_poly, orientation, start_point, 
                stagger_percent, stagger_direction, room_id, 
                sp_includes_grout
            )

            # Store tiles
            if apartment_name not in apartments_data:
                apartments_data[apartment_name] = {
                    'orientation': orientation,
                    'tiles': []
                }
            apartments_data[apartment_name]['tiles'].extend(tiles)

        total_tiles = sum(len(apt_data['tiles']) for apt_name, apt_data in apartments_data.items())
        print(f"✅ Processed {len(apartments_data)} apartments with {total_tiles} total tiles")
        
        return apartments_data
        
    def verify_room_coverage(self, apartments_data, room_df):
        """Verify that tiles completely cover each room"""
        print("\n📋 Verifying room coverage...")
        
        coverage_results = []
        
        for _, room in room_df.iterrows():
            room_id = room['room_id']
            apartment_name = room['apartment_name']
            room_name = room['room_name']
            room_poly = room['polygon']
            room_area = room_poly.area
            
            # Get all tiles for this room
            room_tiles = []
            for apt_name, apt_data in apartments_data.items():
                for tile in apt_data['tiles']:
                    if tile['room_id'] == room_id:
                        room_tiles.append(tile)
            
            # Calculate coverage
            combined_tiles = None
            if room_tiles:
                for tile in room_tiles:
                    poly = tile['layout_polygon']  # Use layout polygon (with grout) for coverage
                    if combined_tiles is None:
                        combined_tiles = poly
                    else:
                        try:
                            combined_tiles = combined_tiles.union(poly)
                        except:
                            try:
                                combined_tiles = combined_tiles.buffer(0).union(poly.buffer(0))
                            except:
                                continue
                
                if combined_tiles:
                    coverage_pct = (combined_tiles.area / room_area) * 100
                else:
                    coverage_pct = 0
            else:
                coverage_pct = 0
            
            coverage_results.append({
                'room_id': room_id,
                'apartment_name': apartment_name,
                'room_name': room_name,
                'room_area': room_area,
                'coverage_pct': coverage_pct,
                'tile_count': len(room_tiles)
            })
        
        coverage_df = pd.DataFrame(coverage_results)
        
        # Check for low coverage
        low_coverage = coverage_df[coverage_df['coverage_pct'] < 99]
        if len(low_coverage) > 0:
            print(f"\n⚠️ Warning: {len(low_coverage)} rooms have less than 99% coverage")
            for _, row in low_coverage.iterrows():
                print(f"  - {row['apartment_name']}-{row['room_name']}: {row['coverage_pct']:.2f}%")
        else:
            print("\n✅ All rooms have at least 99% coverage!")
        
        print(f"\nAverage coverage: {coverage_df['coverage_pct'].mean():.2f}%")
        print(f"Minimum coverage: {coverage_df['coverage_pct'].min():.2f}%")
        
        return coverage_df
                
    def create_grout_visualization(self, apartments_data, room_df=None):
        """Create visualization with white grout lines between tiles"""
        import matplotlib.pyplot as plt
        import io
        import base64
        
        print("\n🎨 Creating visualization with white grout lines...")
        
        plt.figure(figsize=(16, 16))
        
        # Generate apartment colors
        apartment_colors = {apt_name: np.random.rand(3,) for apt_name in apartments_data.keys()}
        
        # Plot room outlines if provided
        if room_df is not None:
            for _, room in room_df.iterrows():
                room_poly = room['polygon']
                if hasattr(room_poly, 'exterior'):
                    x, y = room_poly.exterior.xy
                    plt.plot(x, y, color='black', linewidth=1.5)
                    
                    # Add room label if centroid coordinates are available
                    if 'centroid_x' in room and 'centroid_y' in room:
                        plt.text(room['centroid_x'], room['centroid_y'],
                                f"{room['apartment_name']}-{room['room_name']}",
                                fontsize=10, ha='center', va='center')
        
        # First, plot layout tiles (with grout) in white to create grout lines
        for apt_name, apt_data in apartments_data.items():
            for tile in apt_data['tiles']:
                layout_poly = tile.get('layout_polygon')
                if layout_poly is None:
                    continue
                    
                if isinstance(layout_poly, Polygon) and layout_poly.is_valid and not layout_poly.is_empty:
                    x, y = layout_poly.exterior.xy
                    plt.fill(x, y, color='white', edgecolor='none')
                elif isinstance(layout_poly, MultiPolygon):
                    for part in layout_poly.geoms:
                        if part.is_valid and not part.is_empty:
                            x, y = part.exterior.xy
                            plt.fill(x, y, color='white', edgecolor='none')
        
        # Then, plot actual tiles (factory size) with apartment colors
        rendered_tiles = 0
        for apt_name, apt_data in apartments_data.items():
            color = apartment_colors[apt_name]
            
            for tile in apt_data['tiles']:
                poly = tile.get('polygon')
                if poly is None:
                    continue
                    
                if isinstance(poly, Polygon) and poly.is_valid and not poly.is_empty:
                    x, y = poly.exterior.xy
                    plt.fill(x, y, color=color, edgecolor='black', linewidth=0.2)
                    rendered_tiles += 1
                elif isinstance(poly, MultiPolygon):
                    for part in poly.geoms:
                        if part.is_valid and not part.is_empty:
                            x, y = part.exterior.xy
                            plt.fill(x, y, color=color, edgecolor='black', linewidth=0.2)
                    rendered_tiles += 1
        
        total_tiles = sum(len(apt_data['tiles']) for apt_name, apt_data in apartments_data.items())
        print(f"Total tiles: {total_tiles}, Rendered tiles: {rendered_tiles}")
        
        plt.axis('equal')
        plt.grid(False)
        plt.title(f"Apartment Tile Layout with Grout Lines")
        
        # Add legend
        for apt_name, color in apartment_colors.items():
            plt.plot([], [], color=color, alpha=0.5, linewidth=10, label=apt_name)
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure to buffer and return base64 encoded string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        return {
            'total': total_tiles, 
            'rendered': rendered_tiles,
            'plot': base64.b64encode(buf.read()).decode('utf-8')
        }

    def tile_dataframe(self, tiles):
        # Convert the tiles to a pandas DataFrame for easy analysis
        tile_df = pd.DataFrame(tiles)
        print(f"📋 Converted {len(tile_df)} tiles to DataFrame")
        return tile_df

    def clear_tiles(self):
        self.tiles = []
        print("🗑️ Cleared all tiles")