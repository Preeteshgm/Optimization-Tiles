from flask import Flask, render_template, request, make_response, request, jsonify, session, redirect, url_for, flash
from flask_session import Session
import os
import tempfile
import base64
import io
import matplotlib
matplotlib.use('Agg')
import os
if os.environ.get('VERCEL_ENV'):
    # Ensure the matplotlib cache directory exists
    os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import re  # Add this import for regular expressions
from werkzeug.utils import secure_filename
from shapely.geometry import Polygon, Point, MultiPolygon
from datetime import datetime

# Import processors
from processors.CustomDxfProcessor import CustomDxfProcessor
from processors.VisualizationProcessor import VisualizationProcessor
from processors.RoomClusterProcessor import RoomClusterProcessor
from processors.TileProcessor import TileProcessor
from processors.OptimizationProcessor import OptimizationProcessor
from processors.ExportProcessor import ExportProcessor

# Define the NumpyEncoder class first
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        return super(NumpyEncoder, self).default(obj)

# Create Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# NOW set the JSON encoder after app is defined
app.json_encoder = NumpyEncoder

# Configure server-side sessions
import os
if os.environ.get('VERCEL_ENV'):
    app.config['SESSION_TYPE'] = 'cookie'  # Use cookie-based sessions in production
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_secret_key')  # Set via Vercel env vars
else:
    app.config['SESSION_TYPE'] = 'filesystem'  # Use filesystem in development
    app.config['SESSION_FILE_DIR'] = os.path.join(tempfile.gettempdir(), 'flask_session')

app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
Session(app)  # Initialize the session extension

# Add this back - Upload folder configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize processors as global objects
dxf_processor = CustomDxfProcessor()
cluster_processor = RoomClusterProcessor(eps=5000, min_samples=1)
visualizer = VisualizationProcessor()
tile_processor = TileProcessor()
optimization_processor = OptimizationProcessor()
export_processor = ExportProcessor()

@app.route('/')
def index():
    """Render the landing page"""
    return render_template('landing.html')

@app.route('/step1', methods=['GET', 'POST'])
def step1():
    """Step 1: Load DXF file and process room boundaries"""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'dxf_file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['dxf_file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the file
        try:
            # Set the file path in the processor
            dxf_processor.file_path = filepath
            
            # Load the DXF file
            if not dxf_processor.load_dxf():
                return jsonify({'error': 'Failed to load DXF file'})
            
            # Extract room boundaries
            rooms = dxf_processor.extract_room_boundaries()
            
            # Extract start points
            start_points = dxf_processor.extract_start_points()
            
            # Extract tile sizes
            tile_sizes = dxf_processor.extract_tile_sizes_from_sp()
            
            # Cluster rooms
            room_df = cluster_processor.cluster_rooms(rooms)
            apartment_names = cluster_processor.assign_default_names()
            
            # Generate plots
            room_plot_b64 = visualizer.plot_room_boundaries(rooms, start_points)
            cluster_plot_b64 = visualizer.plot_clusters(room_df)
            
            # Store data in session for next steps (with proper serialization)
            session['rooms_data'] = serialize_rooms(rooms)
            session['start_points_data'] = serialize_start_points(start_points)
            session['tile_sizes'] = tile_sizes
            session['cluster_plot'] = cluster_plot_b64  # Store the cluster plot in session

            # Make sure apartment_name is set in the DataFrame
            if 'apartment_name' not in room_df.columns:
                room_df['apartment_name'] = room_df['apartment_cluster'].apply(
                    lambda x: f"A{x+1}"
                )

            # For room_df, we need to handle the polygon column separately
            room_df_without_polygons = room_df.copy()
            room_df_without_polygons['centroid_x'] = room_df_without_polygons['centroid_x'].astype(float)
            room_df_without_polygons['centroid_y'] = room_df_without_polygons['centroid_y'].astype(float)
            room_df_without_polygons = room_df_without_polygons.drop('polygon', axis=1)
            session['room_df'] = room_df_without_polygons.to_dict('records')
            session['room_polygons'] = serialize_rooms(room_df['polygon'].tolist())
            
            # Return results
            return jsonify({
                'room_count': len(rooms),
                'start_point_count': len(start_points),
                'tile_sizes': tile_sizes,
                'room_plot': room_plot_b64,
                'cluster_plot': cluster_plot_b64
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing DXF file: {str(e)}'})
    
    # GET request - render the template
    return render_template('step1.html')

@app.route('/step2', methods=['GET', 'POST'])
def step2():
    """Step 2: Apartment and Room Naming"""
    if request.method == 'POST':
        data = request.get_json()
        
        if not data or 'apartments' not in data:
            return jsonify({'error': 'Invalid data format'})
        
        try:
            # Get room data from session
            room_df = pd.DataFrame(session.get('room_df', []))
            
            if room_df.empty:
                return jsonify({'error': 'No room data available. Please complete step 1 first.'})
            
            # Update apartment and room names based on user input
            for apt in data['apartments']:
                original_name = apt['original_name']
                new_name = apt['new_name']
                
                # Update apartment names
                room_df.loc[room_df['apartment_name'] == original_name, 'apartment_name'] = new_name
                
                # Update room names
                for room in apt['rooms']:
                    room_id = room['room_id']
                    new_room_name = room['new_name']
                    room_df.loc[room_df['room_id'] == room_id, 'room_name'] = new_room_name
            
            # Store updated data in session
            session['room_df'] = room_df.to_dict('records')
            
            # Generate updated plot
            # We need to rehydrate the DataFrame with polygons for visualization
            room_polygons = deserialize_rooms(session.get('room_polygons', []))
            if len(room_polygons) == len(room_df):
                room_df['polygon'] = room_polygons
                updated_plot_b64 = visualizer.plot_clusters(room_df, use_final_names=True)
            else:
                # If we can't match polygons, use a placeholder image
                updated_plot_b64 = generate_placeholder_image("Updated Room Names", 800, 600)
            
            return jsonify({
                'success': True,
                'updated_plot': updated_plot_b64
            })
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error updating names: {str(e)}'})
    
    # GET request - render the template with apartment data
    try:
        room_df_records = session.get('room_df', [])
        
        if not room_df_records:
            # Redirect to step1 if no data available
            return redirect(url_for('step1'))
        
        # Convert back to DataFrame
        room_df = pd.DataFrame(room_df_records)
        
        # Ensure apartment_name is set
        if 'apartment_name' not in room_df.columns:
            if 'apartment_cluster' in room_df.columns:
                room_df['apartment_name'] = room_df['apartment_cluster'].apply(
                    lambda x: f"A{x+1}"
                )
            else:
                # If there's no way to determine the apartment name, redirect back to step1
                print("Missing apartment_cluster in room_df")
                return redirect(url_for('step1'))
        
        # Prepare data for template
        apartments = []
        for apt_name, group in room_df.groupby('apartment_name'):
            rooms = [
                {
                    'room_id': row['room_id'],
                    'room_name': row['room_name']
                }
                for _, row in group.iterrows()
            ]
            
            apartments.append({
                'apartment_name': apt_name,
                'rooms': rooms
            })
        
        # Include the cluster plot in the response for display alongside form
        cluster_plot = session.get('cluster_plot', '')
        
        return render_template('step2.html', apartments=apartments, cluster_plot=cluster_plot)
    
    except Exception as e:
        # Log the error and redirect to step1
        print(f"Error in step2 GET: {str(e)}")
        import traceback
        traceback.print_exc()
        return redirect(url_for('step1'))

@app.route('/step3', methods=['GET', 'POST'])
def step3():
    """Step 3: Apartment Orientation Input"""
    if request.method == 'POST':
        data = request.get_json()
        
        if not data or 'orientations' not in data:
            return jsonify({'error': 'Invalid data format'})
        
        try:
            # Store orientations in session
            session['orientations'] = data['orientations']
            
            return jsonify({'success': True})
        
        except Exception as e:
            return jsonify({'error': f'Error saving orientations: {str(e)}'})
    
    # GET request - render the template with apartment data
    try:
        room_df = pd.DataFrame(session.get('room_df', []))
        
        if room_df.empty:
            # Redirect to step2 if no data available
            return redirect(url_for('step2'))
        
        # Prepare data for template
        apartments = []
        for apt_name in room_df['apartment_name'].unique():
            apartments.append({
                'apartment_name': apt_name
            })
        
        # Convert to JSON for the template
        apartments_json = json.dumps(apartments)
        
        # Get the updated plot (the one created after saving apartment names)
        # This is better than using the original cluster_plot
        updated_plot = ""
        
        # Try to recreate the plot with current data including room names
        try:
            # Rehydrate the DataFrame with polygons for visualization
            room_polygons = deserialize_rooms(session.get('room_polygons', []))
            if len(room_polygons) == len(room_df):
                room_df['polygon'] = room_polygons
                updated_plot = visualizer.plot_clusters(room_df, use_final_names=True)
            else:
                # If we can't match polygons, use a placeholder image
                updated_plot = generate_placeholder_image("Updated Room Names", 800, 600)
        except Exception as e:
            print(f"Error generating updated plot: {str(e)}")
            # Fall back to the original cluster plot if there's an error
            updated_plot = session.get('cluster_plot', '')
        
        return render_template('step3.html', apartments_json=apartments_json, cluster_plot=updated_plot)
    
    except Exception as e:
        # Log the error and redirect to step2
        print(f"Error in step3 GET: {str(e)}")
        import traceback
        traceback.print_exc()
        return redirect(url_for('step2'))

@app.route('/step4', methods=['GET', 'POST'])
def step4():
    """Step 4: Tile Coverage"""
    if request.method == 'POST':
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Invalid data format'})
        
        try:
            # Get data from session
            rooms_data = session.get('rooms_data', [])
            start_points_data = session.get('start_points_data', [])
            orientations = session.get('orientations', [])
            tile_sizes = session.get('tile_sizes', [])
            
            if not rooms_data or not orientations:
                return jsonify({'error': 'Missing required data. Please complete previous steps first.'})
            
            # Deserialize data
            rooms = deserialize_rooms(rooms_data)
            start_points = deserialize_start_points(start_points_data)
            
            # Extract parameters from the request
            grout_spacing = int(data.get('grout_spacing', 3))
            stagger_percent = float(data.get('stagger_percent', 0))
            stagger_direction = data.get('stagger_direction', 'x')
            
            # Get the room_df for room information
            room_df = pd.DataFrame(session.get('room_df', []))
            room_polygons = deserialize_rooms(session.get('room_polygons', []))
            
            # Add polygons back to room_df for processing
            if len(room_polygons) == len(room_df):
                room_df['polygon'] = room_polygons
            
            # Convert orientations to DataFrame for easy lookup
            orientation_df = pd.DataFrame(orientations)
            
            # Initialize tile processor with default tile size
            default_tile_width = 600
            default_tile_height = 600
            if tile_sizes and len(tile_sizes) > 0:
                default_tile_width, default_tile_height = tile_sizes[0]
            
            # Create tile processor and set grout thickness
            tile_processor = TileProcessor()
            tile_processor.set_tile_size(default_tile_width, default_tile_height)
            tile_processor.set_grout_thickness(grout_spacing)
            
            # Process tiles for each room
            all_tiles = []
            apartments_data = {}
            
            # Process each room based on its apartment orientation
            for _, row in room_df.iterrows():
                room_id = row['room_id']
                room_name = row['room_name']
                apartment_name = row['apartment_name']
                room_poly = row['polygon']
                
                # Find orientation for this apartment
                orientation = 0  # Default orientation
                for _, orient_row in orientation_df.iterrows():
                    if orient_row['apartment_name'] == apartment_name:
                        orientation = orient_row['orientation']
                        break
                
                # Find matching start point for this room
                room_start_point = None
                room_tile_size = (default_tile_width, default_tile_height)  # Default size
                
                for sp in start_points:
                    if 'centroid' in sp and room_poly.contains(Point(sp['centroid'])):
                        room_start_point = Point(sp['centroid'])
                        if 'width' in sp and 'height' in sp:
                            room_tile_size = (sp['width'], sp['height'])
                        break
                
                # Set tile size for this room
                tile_processor.set_tile_size(room_tile_size[0], room_tile_size[1])
                
                # Generate aligned tiles for this room
                tiles = tile_processor.generate_aligned_tile_grid(
                    room_poly, 
                    orientation=orientation,
                    start_point=room_start_point,
                    stagger_percent=stagger_percent,
                    stagger_direction=stagger_direction,
                    room_id=room_id
                )
                
                # Store the tiles by apartment
                if apartment_name not in apartments_data:
                    apartments_data[apartment_name] = {
                        'orientation': orientation,
                        'tiles': []
                    }
                
                apartments_data[apartment_name]['tiles'].extend(tiles)
                all_tiles.extend(tiles)
            
            # Create visualization
            viz_result = tile_processor.create_grout_visualization(apartments_data, room_df)
            tile_plot_b64 = viz_result['plot']
            
            # Store tile data in session
            session['apartments_data'] = serialize_apartments_data(apartments_data)
            session['tile_data'] = {
                'grout_spacing': grout_spacing,
                'stagger_percent': stagger_percent,
                'stagger_direction': stagger_direction,
                'tile_count': viz_result['total']
            }
            
            return jsonify({
                'success': True,
                'tile_plot': tile_plot_b64,
                'tile_count': viz_result['total']
            })
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error generating tile layout: {str(e)}'})
    
    # GET request - render the template
    try:
        # Check if required data exists
        if 'orientations' not in session:
            return redirect(url_for('step3'))
        
        return render_template('step4.html')
    
    except Exception as e:
        print(f"Error in step4 GET: {str(e)}")
        return redirect(url_for('step3'))

def serialize_apartments_data(apartments_data):
    """Serialize apartment data with tiles for storage in session"""
    serialized_data = {}
    
    for apt_name, apt_data in apartments_data.items():
        serialized_tiles = []
        
        for tile in apt_data['tiles']:
            # Create a base serializable tile without polygons
            serialized_tile = {k: v for k, v in tile.items() if k != 'polygon' and k != 'layout_polygon'}
            
            # Serialize each tile's polygon - handle both Polygon and MultiPolygon
            if 'polygon' in tile and tile['polygon'] is not None:
                polygon = tile['polygon']
                if isinstance(polygon, Polygon):
                    serialized_tile['polygon_coords'] = list(polygon.exterior.coords)
                    serialized_tile['polygon_type'] = 'Polygon'
                    # Also serialize interior rings if present
                    if len(polygon.interiors) > 0:
                        serialized_tile['polygon_interiors'] = [list(interior.coords) 
                                                            for interior in polygon.interiors]
                elif isinstance(polygon, MultiPolygon):
                    # For MultiPolygon, store coordinates of each part
                    serialized_tile['polygon_parts'] = []
                    for part in polygon.geoms:
                        geom_data = {'exterior': list(part.exterior.coords)}
                        if len(part.interiors) > 0:
                            geom_data['interiors'] = [list(interior.coords) 
                                                    for interior in part.interiors]
                        serialized_tile['polygon_parts'].append(geom_data)
                    serialized_tile['polygon_type'] = 'MultiPolygon'
            
            # Serialize layout_polygon if it exists
            if 'layout_polygon' in tile and tile['layout_polygon'] is not None:
                layout_polygon = tile['layout_polygon']
                if isinstance(layout_polygon, Polygon):
                    serialized_tile['layout_polygon_coords'] = list(layout_polygon.exterior.coords)
                    serialized_tile['layout_polygon_type'] = 'Polygon'
                    # Serialize interior rings if present
                    if len(layout_polygon.interiors) > 0:
                        serialized_tile['layout_polygon_interiors'] = [list(interior.coords) 
                                                                   for interior in layout_polygon.interiors]
                elif isinstance(layout_polygon, MultiPolygon):
                    # For MultiPolygon, store coordinates of each part
                    serialized_tile['layout_polygon_parts'] = []
                    for part in layout_polygon.geoms:
                        geom_data = {'exterior': list(part.exterior.coords)}
                        if len(part.interiors) > 0:
                            geom_data['interiors'] = [list(interior.coords) 
                                                    for interior in part.interiors]
                        serialized_tile['layout_polygon_parts'].append(geom_data)
                    serialized_tile['layout_polygon_type'] = 'MultiPolygon'
            
            serialized_tiles.append(serialized_tile)
        
        serialized_data[apt_name] = {
            'orientation': apt_data['orientation'],
            'tiles': serialized_tiles
        }
    
    return serialized_data


def deserialize_apartments_data(serialized_data):
    """Deserialize apartment data from session storage"""
    apartments_data = {}
    
    for apt_name, apt_data in serialized_data.items():
        tiles = []
        
        for tile_data in apt_data['tiles']:
            # Create a base tile without polygons
            tile = {k: v for k, v in tile_data.items() 
                  if k not in ['polygon_coords', 'polygon_parts', 'polygon_type',
                              'layout_polygon_coords', 'layout_polygon_parts', 'layout_polygon_type',
                              'polygon_interiors', 'layout_polygon_interiors']}
            
            # Deserialize the polygon
            if 'polygon_type' in tile_data:
                if tile_data['polygon_type'] == 'Polygon' and 'polygon_coords' in tile_data:
                    # Handle interior rings if present
                    if 'polygon_interiors' in tile_data:
                        tile['polygon'] = Polygon(
                            tile_data['polygon_coords'], 
                            [interior for interior in tile_data['polygon_interiors']]
                        )
                    else:
                        tile['polygon'] = Polygon(tile_data['polygon_coords'])
                elif tile_data['polygon_type'] == 'MultiPolygon' and 'polygon_parts' in tile_data:
                    # Create polygons for each part
                    polygons = []
                    for part in tile_data['polygon_parts']:
                        if isinstance(part, dict) and 'exterior' in part:
                            if 'interiors' in part:
                                poly = Polygon(part['exterior'], part['interiors'])
                            else:
                                poly = Polygon(part['exterior'])
                        else:
                            # Backward compatibility
                            poly = Polygon(part)
                        polygons.append(poly)
                    tile['polygon'] = MultiPolygon(polygons)
            
            # Deserialize the layout polygon
            if 'layout_polygon_type' in tile_data:
                if tile_data['layout_polygon_type'] == 'Polygon' and 'layout_polygon_coords' in tile_data:
                    # Handle interior rings if present
                    if 'layout_polygon_interiors' in tile_data:
                        tile['layout_polygon'] = Polygon(
                            tile_data['layout_polygon_coords'], 
                            [interior for interior in tile_data['layout_polygon_interiors']]
                        )
                    else:
                        tile['layout_polygon'] = Polygon(tile_data['layout_polygon_coords'])
                elif tile_data['layout_polygon_type'] == 'MultiPolygon' and 'layout_polygon_parts' in tile_data:
                    # Create polygons for each part
                    polygons = []
                    for part in tile_data['layout_polygon_parts']:
                        if isinstance(part, dict) and 'exterior' in part:
                            if 'interiors' in part:
                                poly = Polygon(part['exterior'], part['interiors'])
                            else:
                                poly = Polygon(part['exterior'])
                        else:
                            # Backward compatibility
                            poly = Polygon(part)
                        polygons.append(poly)
                    tile['layout_polygon'] = MultiPolygon(polygons)
            
            tiles.append(tile)
        
        apartments_data[apt_name] = {
            'orientation': apt_data['orientation'],
            'tiles': tiles
        }
    
    return apartments_data


def validate_tiles_data(apartments_data, step_name=""):
    """Validate tiles data for debugging"""
    total_tiles = 0
    total_with_polygon = 0
    total_with_classification = 0
    total_multipolygons = 0
    total_split_tiles = 0
    tiles_by_classification = {
        'full': 0,
        'irregular': 0,
        'cut_x': 0,
        'cut_y': 0,
        'all_cut': 0,
        'split': 0,
        'split_cut': 0,
        'unknown': 0
    }
    
    for apt_name, apt_data in apartments_data.items():
        print(f"Apartment {apt_name}: {len(apt_data['tiles'])} tiles, orientation: {apt_data['orientation']}")
        for i, tile in enumerate(apt_data['tiles']):
            total_tiles += 1
            
            # Check polygon
            if 'polygon' in tile and tile['polygon'] is not None:
                total_with_polygon += 1
                if isinstance(tile['polygon'], MultiPolygon):
                    total_multipolygons += 1
            
            # Check if this is a split tile
            if 'is_split' in tile and tile['is_split']:
                total_split_tiles += 1
            
            # Check classification
            if 'classification' in tile:
                total_with_classification += 1
                class_type = tile['classification']
                if class_type in tiles_by_classification:
                    tiles_by_classification[class_type] += 1
                else:
                    tiles_by_classification['unknown'] += 1
            
            # Check type (for unclassified tiles)
            elif 'type' in tile:
                type_value = tile['type']
                if type_value in tiles_by_classification:
                    tiles_by_classification[type_value] += 1
                else:
                    tiles_by_classification['unknown'] += 1
                    
            # Debug first 5 tiles of each apartment
            if i < 5:
                tile_type = tile.get('type', 'unknown')
                is_split = tile.get('is_split', False)
                part_index = tile.get('part_index', None)
                if 'polygon' in tile:
                    poly_type = type(tile['polygon']).__name__
                else:
                    poly_type = "None"
                print(f"  Tile {i}: type={tile_type}, is_split={is_split}, part_index={part_index}, polygon_type={poly_type}")
    
    print(f"\n=== {step_name} Validation ===")
    print(f"  Total tiles: {total_tiles}")
    if total_tiles > 0:
        print(f"  With polygon: {total_with_polygon} ({total_with_polygon/total_tiles*100:.1f}%)")
        print(f"  With classification: {total_with_classification} ({total_with_classification/total_tiles*100:.1f}%)")
        print(f"  MultiPolygons: {total_multipolygons} ({total_multipolygons/total_tiles*100:.1f}%)")
        print(f"  Split tiles: {total_split_tiles} ({total_split_tiles/total_tiles*100:.1f}%)")
    else:
        print("  No tiles found!")
    print(f"  Classification/type breakdown: {tiles_by_classification}")
    
    return {
        "total_tiles": total_tiles,
        "total_with_polygon": total_with_polygon,
        "total_with_classification": total_with_classification,
        "total_multipolygons": total_multipolygons,
        "total_split_tiles": total_split_tiles,
        "tiles_by_classification": tiles_by_classification
    }


def check_for_multipolygons(apartments_data):
    """Count MultiPolygons in the data"""
    multipolygon_count = 0
    for apt_name, apt_data in apartments_data.items():
        for tile_idx, tile in enumerate(apt_data['tiles']):
            if 'polygon' in tile and isinstance(tile['polygon'], MultiPolygon):
                multipolygon_count += 1
                parts = list(tile['polygon'].geoms)
                print(f"MultiPolygon found in {apt_name}, tile {tile_idx} with {len(parts)} parts")
    
    print(f"Found {multipolygon_count} MultiPolygons")
    return multipolygon_count

def detect_and_handle_multipolygons(apartments_data):
    """Check and handle any MultiPolygons before classification"""
    multipolygon_count = 0
    for apt_name, apt_data in apartments_data.items():
        for tile_idx, tile in enumerate(apt_data['tiles']):
            if isinstance(tile['polygon'], MultiPolygon):
                multipolygon_count += 1
                # Keep only the largest part to ensure consistent classification
                parts = list(tile['polygon'].geoms)
                areas = [part.area for part in parts]
                largest_part_idx = areas.index(max(areas))
                tile['polygon'] = parts[largest_part_idx]
                
    if multipolygon_count > 0:
        print(f"Fixed {multipolygon_count} MultiPolygon tiles for classification")
    return apartments_data

@app.route('/step5', methods=['GET', 'POST'])
def step5():
    """Step 5: Tile Classification"""
    if request.method == 'POST':
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Invalid data format'})
        
        try:
            # Check if tile data exists in session
            if 'apartments_data' not in session:
                return jsonify({'error': 'No tile data found. Please complete step 4 first.'})
            
            # Get data from session
            apartments_data = deserialize_apartments_data(session.get('apartments_data', {}))
            
            # Debug validation before any modifications
            before_validation = validate_tiles_data(apartments_data, "Before MultiPolygon handling")
            
            # Check for MultiPolygons
            multipolygon_count = check_for_multipolygons(apartments_data)
            
            # MANDATORY: Split MultiPolygons into individual tiles
            if multipolygon_count > 0:
                print("Splitting MultiPolygons into individual tiles (MANDATORY step)...")
                apartments_data = split_multipolygons_into_individual_tiles(apartments_data)
                
                # Validate after splitting
                after_splitting = validate_tiles_data(apartments_data, "After MultiPolygon splitting")
            
            # Get room_df for room information
            room_df = pd.DataFrame(session.get('room_df', []))
            room_polygons = deserialize_rooms(session.get('room_polygons', []))
            
            # Add polygons back to room_df for processing
            if len(room_polygons) == len(room_df):
                room_df['polygon'] = room_polygons
            
            # Whether tiles have pattern
            has_pattern = data.get('has_pattern', False)
            
            # Classify tiles
            tiles_df, full_tiles, irregular_tiles, cut_x_tiles, cut_y_tiles, all_cut_tiles, cut_x_df, cut_y_df, all_cut_df, stats = optimize_tile_classification(
                apartments_data, room_df, has_pattern)
            
            # Create visualization
            classification_plot_b64 = visualize_classification(tiles_df, room_df, has_pattern)
            
            # Store classification data in session
            # Note: We store the updated apartments_data with classification info
            session['classification_data'] = {
                'has_pattern': has_pattern,
                'total_tiles': stats['total_tiles'],
                'full_tiles': stats['full_tiles'],
                'irregular_tiles': stats['irregular_tiles'],
                'cut_x_tiles': stats['cut_x_tiles'],
                'cut_y_tiles': stats['cut_y_tiles'],
                'all_cut_tiles': stats['all_cut_tiles'],
                'classified_apartments_data': serialize_apartments_data(apartments_data)
            }
            
            # Prepare statistics for response
            stats_data = []
            stats_data.append({'type': 'Full', 'count': stats['full_tiles'], 
                             'percentage': round(stats['full_tiles'] / stats['total_tiles'] * 100, 1)})
            stats_data.append({'type': 'Irregular', 'count': stats['irregular_tiles'], 
                             'percentage': round(stats['irregular_tiles'] / stats['total_tiles'] * 100, 1)})
            
            if has_pattern:
                stats_data.append({'type': 'Cut X', 'count': stats['cut_x_tiles'], 
                                 'percentage': round(stats['cut_x_tiles'] / stats['total_tiles'] * 100, 1)})
                stats_data.append({'type': 'Cut Y', 'count': stats['cut_y_tiles'], 
                                 'percentage': round(stats['cut_y_tiles'] / stats['total_tiles'] * 100, 1)})
                
                # Cut X types data
                cutx_types = []
                for _, row in stats['cut_x_types'].iterrows():
                    cutx_types.append({
                        'id': len(cutx_types) + 1,
                        'cut_side': int(row['cut_side']),
                        'count': int(row['count'])
                    })
                
                # Cut Y types data
                cuty_types = []
                for _, row in stats['cut_y_types'].iterrows():
                    cuty_types.append({
                        'id': len(cuty_types) + 1,
                        'cut_side': int(row['cut_side']),
                        'count': int(row['count'])
                    })
                
                # Empty all cut types
                all_cut_types = []
                
            else:
                stats_data.append({'type': 'All Cut', 'count': stats['all_cut_tiles'], 
                                 'percentage': round(stats['all_cut_tiles'] / stats['total_tiles'] * 100, 1)})
                
                # All cut types data
                all_cut_types = []
                for _, row in stats['all_cut_types'].iterrows():
                    all_cut_types.append({
                        'id': len(all_cut_types) + 1,
                        'cut_dim': int(row['cut_dim']),
                        'count': int(row['count'])
                    })
                
                # Empty cut X and Y types
                cutx_types = []
                cuty_types = []
            
            return jsonify({
                'success': True,
                'classification_plot': classification_plot_b64,
                'has_pattern': has_pattern,
                'stats': stats_data,
                'cut_x_types': cutx_types,
                'cut_y_types': cuty_types,
                'all_cut_types': all_cut_types,
                'debug_info': {
                    'multipolygons_found': multipolygon_count,
                    'validation_before': before_validation,
                    'validation_after_splitting': after_splitting if multipolygon_count > 0 else None
                }
            })
            
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            print(traceback_str)
            return jsonify({'error': f'Error classifying tiles: {str(e)}', 'traceback': traceback_str})
    
    # GET request - render the template
    try:
        # Check if required data exists
        if 'tile_data' not in session:
            return redirect(url_for('step4'))
        
        return render_template('step5.html')
    
    except Exception as e:
        print(f"Error in step5 GET: {str(e)}")
        return redirect(url_for('step4'))

def split_multipolygons_into_individual_tiles(apartments_data):
    """
    Split each MultiPolygon into separate individual tiles.
    This is a mandatory step for proper tile processing.
    
    Parameters:
    - apartments_data: Dictionary of apartment data with tiles
    
    Returns:
    - Updated apartments_data with MultiPolygons split into individual tiles
    """
    print("\n🔪 Splitting MultiPolygons into individual tiles...")
    
    # Create a copy of the apartments_data to avoid modifying the original
    split_data = {apt_name: {'orientation': apt_data['orientation'], 
                             'tiles': apt_data['tiles'].copy()} 
                 for apt_name, apt_data in apartments_data.items()}
    
    # Track the original and split tiles
    split_results = []
    total_new_tiles = 0
    multipolygons_found = 0
    
    # Process each apartment
    for apt_name, apt_data in split_data.items():
        # We need to create a new tiles list to avoid modifying during iteration
        new_tiles_list = []
        original_tiles_to_remove = []
        
        # First pass: identify MultiPolygons and prepare split tiles
        for tile_idx, tile in enumerate(apt_data['tiles']):
            if isinstance(tile['polygon'], MultiPolygon):
                multipolygons_found += 1
                room_id = tile['room_id']
                tile_id = f"{apt_name}-R{room_id}-T{tile_idx}"
                
                # Get the MultiPolygon parts
                parts = list(tile['polygon'].geoms)
                
                # Only process if there are multiple parts
                if len(parts) > 1:
                    print(f"  Splitting MultiPolygon {tile_id} into {len(parts)} separate tiles")
                    
                    # Mark this tile for removal later
                    original_tiles_to_remove.append(tile_idx)
                    
                    # Create a new tile for each part
                    for part_idx, part in enumerate(parts):
                        # Create a new tile based on the original
                        new_tile = tile.copy()
                        new_tile['polygon'] = part
                        new_tile['type'] = 'split' if new_tile.get('type', '') == 'full' else 'split_cut'
                        new_tile['original_area'] = tile['polygon'].area if 'polygon' in tile else 0
                        new_tile['part_area'] = part.area
                        new_tile['part_percent'] = (part.area / tile['polygon'].area * 100) if 'polygon' in tile and tile['polygon'].area > 0 else 0
                        new_tile['is_split'] = True
                        new_tile['part_index'] = part_idx
                        new_tile['original_tile_index'] = tile_idx  # Add reference to original
                        
                        # Update the centroid if needed
                        if hasattr(part, 'centroid'):
                            new_tile['centroid'] = (part.centroid.x, part.centroid.y)
                        
                        # Add to new tiles list
                        new_tiles_list.append(new_tile)
                    
                    # Record the split result
                    split_results.append({
                        'tile_id': tile_id,
                        'original_parts': len(parts),
                        'largest_part_percent': (max(part.area for part in parts) / tile['polygon'].area * 100) if 'polygon' in tile and tile['polygon'].area > 0 else 0,
                        'new_tiles': len(parts),
                        'area_preserved': 100.0  # We keep all the area
                    })
                    
                    total_new_tiles += len(parts)
            else:
                # Keep this tile as-is
                new_tiles_list.append(tile)
        
        # Update the apartment's tiles with the new list
        apt_data['tiles'] = new_tiles_list
    
    # Display results
    if multipolygons_found > 0:
        print(f"\n✅ Split completed: {multipolygons_found} MultiPolygons found, created {total_new_tiles} new tiles")
    else:
        print("\n✅ No MultiPolygons found, no splitting needed")
    
    return split_data

def optimize_tile_classification(apartments_data, final_room_df, has_pattern=False):
    """
    Optimized and simplified tile classification to handle both pattern and no-pattern cases
    
    Parameters:
    - apartments_data: Dictionary of apartment data
    - final_room_df: DataFrame with room data
    - has_pattern: Boolean indicating if tiles have pattern
    """
    print(f"🔄 Classifying tiles with {'pattern consideration' if has_pattern else 'flat tile (no pattern) assumption'}...")
    
    # Create a list to hold all tile data
    all_tiles = []
    
    # Track total tiles for verification
    total_tiles = 0
    
    # Calculate grout thickness from the first tile
    grout_thickness = 3  # Default if not found
    for apt_name, apt_data in apartments_data.items():
        if apt_data['tiles']:
            first_tile = apt_data['tiles'][0]
            if 'width' in first_tile and 'actual_tile_width' in first_tile:
                grout_thickness = first_tile['width'] - first_tile['actual_tile_width']
                break
    
    print(f"✅ Using grout thickness: {grout_thickness} mm")
    
    # Process each apartment and its tiles
    for apt_name, apt_data in apartments_data.items():
        orientation = apt_data['orientation']
        total_tiles += len(apt_data['tiles'])
        
        for tile_idx, tile in enumerate(apt_data['tiles']):
            # Get room information
            room_id = tile['room_id']
            
            # Find room in room_df
            room_row = final_room_df[final_room_df['room_id'] == room_id]
            if not room_row.empty:
                room_name = room_row['room_name'].values[0]
            else:
                room_name = f"Room {room_id}"
            
            # Get polygon and measurements
            polygon = tile.get('polygon')
            if polygon is None:
                continue
                
            # Get dimensions
            width = tile.get('width', 0)
            height = tile.get('height', 0)
            actual_width = tile.get('actual_tile_width', width - grout_thickness)
            actual_height = tile.get('actual_tile_height', height - grout_thickness)
                
            # Measured dimensions
            if isinstance(polygon, Polygon):
                minx, miny, maxx, maxy = polygon.bounds
                measured_width = maxx - minx
                measured_height = maxy - miny
            elif isinstance(polygon, MultiPolygon):
                minx, miny, maxx, maxy = polygon.bounds
                measured_width = maxx - minx
                measured_height = maxy - miny
            else:
                continue
            
            # Count sides
            if isinstance(polygon, Polygon):
                sides = len(list(polygon.exterior.coords)) - 1
            elif isinstance(polygon, MultiPolygon):
                largest = max(polygon.geoms, key=lambda p: p.area)
                sides = len(list(largest.exterior.coords)) - 1
            else:
                sides = 0
            
            # Apply apartment orientation to determine expected dimensions
            if orientation == 90:
                expected_width = actual_height
                expected_height = actual_width
            else:
                expected_width = actual_width
                expected_height = actual_height
            
            # Classification tolerance (2% for better capture of cut tiles)
            tolerance = 0.02  # Increased from 0.01 to 0.02
            
            # Check if the tile is full-sized in either dimension
            width_ratio = measured_width / expected_width if expected_width > 0 else 0
            height_ratio = measured_height / expected_height if expected_height > 0 else 0
            
            is_full_width = abs(1.0 - width_ratio) <= tolerance
            is_full_height = abs(1.0 - height_ratio) <= tolerance
            
            # Classification logic with improved orientation handling
            if sides > 4:
                # Irregular tiles
                classification = 'irregular'
                cut_side = None
            elif is_full_width and is_full_height:
                # Full tiles
                classification = 'full'
                cut_side = None
            elif has_pattern:
                # With pattern: separate cut_x and cut_y
                # Adjust classification based on apartment orientation
                if orientation == 90:
                    # In 90-degree rotated apartments, X and Y are swapped
                    if is_full_width:
                        # Full width in rotated space means full Y in original space
                        classification = 'cut_x'  # Cut in X direction in original space
                        cut_side = round(measured_height)
                    elif is_full_height:
                        # Full height in rotated space means full X in original space
                        classification = 'cut_y'  # Cut in Y direction in original space
                        cut_side = round(measured_width)
                    else:
                        # Cut in both directions - classify based on which has greater deviation
                        width_deviation = abs(1.0 - width_ratio)
                        height_deviation = abs(1.0 - height_ratio)
                        
                        if width_deviation >= height_deviation:
                            classification = 'cut_y'  # Swapped for 90-degree orientation
                            cut_side = round(measured_width)
                        else:
                            classification = 'cut_x'  # Swapped for 90-degree orientation
                            cut_side = round(measured_height)
                else:
                    # Normal orientation (0, 180, 270 etc.) - original logic
                    if is_full_width:
                        # Full width, cut height
                        classification = 'cut_y'
                        cut_side = round(measured_height)
                    elif is_full_height:
                        # Full height, cut width
                        classification = 'cut_x'
                        cut_side = round(measured_width)
                    else:
                        # Cut in both directions - classify based on which has greater deviation
                        width_deviation = abs(1.0 - width_ratio)
                        height_deviation = abs(1.0 - height_ratio)
                        
                        if width_deviation >= height_deviation:
                            classification = 'cut_x'
                            cut_side = round(measured_width)
                        else:
                            classification = 'cut_y'
                            cut_side = round(measured_height)
            else:
                # No pattern: all cuts in one list
                # FIXED: More inclusive logic for all_cut classification
                if not (is_full_width and is_full_height):
                    # Any tile that's not both full width AND full height is a cut tile
                    classification = 'all_cut'
                    
                    # Determine cut dimension based on which dimension is cut
                    if is_full_width and not is_full_height:
                        # Cut in height (Y direction)
                        cut_side = round(measured_height)
                    elif is_full_height and not is_full_width:
                        # Cut in width (X direction)
                        cut_side = round(measured_width)
                    else:
                        # Cut in both directions - use the smaller dimension as the cut
                        # This represents the limiting dimension that defines the usable piece
                        cut_side = min(round(measured_width), round(measured_height))
                else:
                    # This should not happen as we already checked for full tiles above
                    classification = 'full'
                    cut_side = None
            
            # Store all tile data
            all_tiles.append({
                'apartment_name': apt_name,
                'room_id': room_id,
                'room_name': room_name,
                'tile_index': tile_idx,
                'polygon': polygon,
                'orientation': orientation,
                'measured_width': measured_width,
                'measured_height': measured_height,
                'width': width,
                'height': height,
                'actual_width': actual_width,
                'actual_height': actual_height,
                'is_full_width': is_full_width,
                'is_full_height': is_full_height,
                'classification': classification,
                'sides': sides,
                'cut_side': cut_side if classification in ['cut_x', 'cut_y', 'all_cut'] else None
            })
            
            # Update the original tile with classification info
            tile['classification'] = classification
            tile['cut_side'] = cut_side
    
    # Convert to DataFrame
    tiles_df = pd.DataFrame(all_tiles)
    
    # Verify all tiles were preserved
    print(f"✅ Processed {len(tiles_df)} tiles out of {total_tiles} total")
    
    # Create classification subsets
    full_tiles = tiles_df[tiles_df['classification'] == 'full'].copy()
    irregular_tiles = tiles_df[tiles_df['classification'] == 'irregular'].copy()
    
    # Generate tile summary data
    if has_pattern:
        # With pattern: separate cut_x and cut_y
        cut_x_tiles = tiles_df[tiles_df['classification'] == 'cut_x'].copy()
        cut_y_tiles = tiles_df[tiles_df['classification'] == 'cut_y'].copy()
        all_cut_tiles = pd.DataFrame()  # Empty
        
        # Generate cut type summaries
        cut_x_types = cut_x_tiles.groupby('cut_side').size().reset_index(name='count')
        cut_y_types = cut_y_tiles.groupby('cut_side').size().reset_index(name='count')
        
        # Sort by cut side
        if not cut_x_types.empty:
            cut_x_types = cut_x_types.sort_values('cut_side')
        if not cut_y_types.empty:
            cut_y_types = cut_y_types.sort_values('cut_side')
        
        all_cut_types = pd.DataFrame()  # Empty
        
    else:
        # No pattern: all cuts in one list
        all_cut_tiles = tiles_df[tiles_df['classification'] == 'all_cut'].copy()
        cut_x_tiles = pd.DataFrame()  # Empty
        cut_y_tiles = pd.DataFrame()  # Empty
        
        # Generate all cut types summary
        all_cut_types = all_cut_tiles.groupby('cut_side').size().reset_index(name='count')
        
        # Sort by cut side
        if not all_cut_types.empty:
            all_cut_types = all_cut_types.sort_values('cut_side')
            all_cut_types.rename(columns={'cut_side': 'cut_dim'}, inplace=True)
        
        cut_x_types = pd.DataFrame()  # Empty
        cut_y_types = pd.DataFrame()  # Empty
    
    # Calculate statistics
    stats = {
        'total_tiles': len(tiles_df),
        'full_tiles': len(full_tiles),
        'irregular_tiles': len(irregular_tiles),
        'cut_x_tiles': len(cut_x_tiles),
        'cut_y_tiles': len(cut_y_tiles),
        'all_cut_tiles': len(all_cut_tiles),
        'cut_x_types': cut_x_types,
        'cut_y_types': cut_y_types,
        'all_cut_types': all_cut_types,
        'grout_thickness': grout_thickness,
        'has_pattern': has_pattern
    }
    
    # Debug output for no-pattern mode
    if not has_pattern:
        print(f"\n📊 No-Pattern Classification Debug:")
        print(f"  Total tiles: {len(tiles_df)}")
        print(f"  Full tiles: {len(full_tiles)}")
        print(f"  All cut tiles: {len(all_cut_tiles)}")
        print(f"  Irregular tiles: {len(irregular_tiles)}")
        
        # Show some examples of all_cut tiles
        if not all_cut_tiles.empty:
            print(f"  Sample all_cut tiles:")
            for idx, (_, tile) in enumerate(all_cut_tiles.head(3).iterrows()):
                print(f"    {idx+1}. Apt: {tile['apartment_name']}, Room: {tile['room_name']}, "
                      f"Measured: {tile['measured_width']:.1f}x{tile['measured_height']:.1f}, "
                      f"Expected: {tile['actual_width']:.1f}x{tile['actual_height']:.1f}, "
                      f"Cut side: {tile['cut_side']}")
    
    # The following dataframes would be generated for export in a real implementation
    # but we don't need them for web display
    cut_x_df = pd.DataFrame()
    cut_y_df = pd.DataFrame()
    all_cut_df = pd.DataFrame()
    
    return tiles_df, full_tiles, irregular_tiles, cut_x_tiles, cut_y_tiles, all_cut_tiles, cut_x_df, cut_y_df, all_cut_df, stats

def debug_tile_classification(tiles_df, has_pattern=False):
    """
    Debug function to analyze tile classification results
    """
    print(f"\n🔍 CLASSIFICATION DEBUG REPORT:")
    print(f"Pattern mode: {has_pattern}")
    print(f"Total tiles analyzed: {len(tiles_df)}")
    
    # Classification breakdown
    classification_counts = tiles_df['classification'].value_counts()
    print(f"\nClassification breakdown:")
    for classification, count in classification_counts.items():
        percentage = (count / len(tiles_df)) * 100
        print(f"  {classification}: {count} ({percentage:.1f}%)")
    
    # For no-pattern mode, show detailed all_cut analysis
    if not has_pattern and 'all_cut' in classification_counts:
        all_cut_tiles = tiles_df[tiles_df['classification'] == 'all_cut']
        
        print(f"\n📏 All-Cut Tiles Analysis:")
        print(f"  Total all-cut tiles: {len(all_cut_tiles)}")
        
        # Group by cut dimension
        cut_dimensions = all_cut_tiles['cut_side'].value_counts().sort_index()
        print(f"  Cut dimensions distribution:")
        for dim, count in cut_dimensions.items():
            print(f"    {dim}mm: {count} tiles")
        
        # Show tiles that might have been missed (full dimensions but classified as cut)
        print(f"\n🔍 Sample all-cut tiles (first 5):")
        for idx, (_, tile) in enumerate(all_cut_tiles.head(5).iterrows()):
            width_ratio = tile['measured_width'] / tile['actual_width'] if tile['actual_width'] > 0 else 0
            height_ratio = tile['measured_height'] / tile['actual_height'] if tile['actual_height'] > 0 else 0
            
            print(f"    {idx+1}. {tile['apartment_name']}-{tile['room_name']}")
            print(f"       Measured: {tile['measured_width']:.1f} x {tile['measured_height']:.1f}")
            print(f"       Expected: {tile['actual_width']:.1f} x {tile['actual_height']:.1f}")
            print(f"       Ratios: W={width_ratio:.3f}, H={height_ratio:.3f}")
            print(f"       Cut side: {tile['cut_side']}mm")
            print(f"       Full W: {tile['is_full_width']}, Full H: {tile['is_full_height']}")
    
    # Check for any tiles that might be misclassified
    potential_issues = tiles_df[
        (tiles_df['classification'] == 'full') & 
        ((abs(tiles_df['measured_width'] - tiles_df['actual_width']) > 5) | 
         (abs(tiles_df['measured_height'] - tiles_df['actual_height']) > 5))
    ]
    
    if not potential_issues.empty:
        print(f"\n⚠️  Potential classification issues ({len(potential_issues)} tiles):")
        for idx, (_, tile) in enumerate(potential_issues.head(3).iterrows()):
            print(f"    {idx+1}. {tile['apartment_name']}-{tile['room_name']} classified as '{tile['classification']}'")
            print(f"       Measured: {tile['measured_width']:.1f} x {tile['measured_height']:.1f}")
            print(f"       Expected: {tile['actual_width']:.1f} x {tile['actual_height']:.1f}")
    
    return classification_counts

def visualize_classification(tiles_df, final_room_df, has_pattern=False, with_grout=True):
    """Optimized visualization of classified tiles"""
    print(f"\n🎨 Visualizing classified tiles...")
    
    plt.figure(figsize=(16, 16))
    
    # Plot room outlines
    for _, room in final_room_df.iterrows():
        room_poly = room['polygon']
        if hasattr(room_poly, 'exterior'):
            x, y = room_poly.exterior.xy
            plt.plot(x, y, color='black', linewidth=1.5, alpha=0.8)
            
            # Add room label if centroid coordinates are available
            if 'centroid_x' in room and 'centroid_y' in room:
                plt.text(room['centroid_x'], room['centroid_y'], 
                        f"{room['apartment_name']}-{room['room_name']}", 
                        fontsize=10, ha='center', va='center', 
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # Define colors for different classifications
    colors = {
        'full': 'green',
        'irregular': 'blue',
        'cut_x': 'orange',
        'cut_y': 'red',
        'all_cut': 'purple',
        'unknown': 'gray'
    }
    
    # Track counts for legend
    classification_counts = {cls: 0 for cls in colors}
    classification_counts['total'] = 0
    
    # Plot each tile with its classification color
    for _, tile in tiles_df.iterrows():
        classification = tile['classification']
        color = colors.get(classification, 'gray')
        poly = tile['polygon']
        
        try:
            if isinstance(poly, Polygon) and poly.is_valid and not poly.is_empty:
                x, y = poly.exterior.xy
                plt.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                classification_counts[classification] += 1
                classification_counts['total'] += 1
            elif isinstance(poly, MultiPolygon):
                for part in poly.geoms:
                    if part.is_valid and not part.is_empty:
                        x, y = part.exterior.xy
                        plt.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                classification_counts[classification] += 1
                classification_counts['total'] += 1
        except Exception as e:
            print(f"Error plotting tile: {e}")
    
    # Add legend with counts
    legend_elements = []
    for cls, color in colors.items():
        count = classification_counts.get(cls, 0)
        if count > 0:  # Only show classes that have tiles
            from matplotlib.patches import Patch
            legend_elements.append(
                Patch(facecolor=color, alpha=0.7, edgecolor='black', 
                      label=f"{cls.capitalize()} ({count}, {count/classification_counts['total']*100:.1f}%)")
            )
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(f"Tiles Classified by Type with {'Pattern' if has_pattern else 'No Pattern'} {' (with Grout Lines)' if with_grout else ''}")
    plt.axis('equal')
    plt.grid(False)
    plt.tight_layout()
    
    # Save figure to buffer and return base64 encoded string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/step6', methods=['GET', 'POST'])
def step6():
    """Step 6: Identify Small Cut Tiles"""
    if request.method == 'POST':
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Invalid data format'})
        
        try:
            # Check if classification data exists
            if 'classification_data' not in session:
                return jsonify({'error': 'No classification data found. Please complete step 5 first.'})
            
            # Get data from session
            classification_data = session.get('classification_data', {})
            apartments_data = deserialize_apartments_data(classification_data.get('classified_apartments_data', {}))
            has_pattern = classification_data.get('has_pattern', False)
            
            # Validate data
            validate_tiles_data(apartments_data, "Step 6 Start")
            
            # Get room_df for room information
            room_df = pd.DataFrame(session.get('room_df', []))
            room_polygons = deserialize_rooms(session.get('room_polygons', []))
            
            # Add polygons back to room_df for processing
            if len(room_polygons) == len(room_df):
                room_df['polygon'] = room_polygons
            
            # Get size threshold
            size_threshold = float(data.get('size_threshold', 10))
            exclude_small_cuts = data.get('exclude_small_cuts', True)
            
            # First, regenerate the tiles_df from the apartments_data
            all_tiles = []
            for apt_name, apt_data in apartments_data.items():
                for tile_idx, tile in enumerate(apt_data['tiles']):
                    if 'polygon' not in tile or tile['polygon'] is None:
                        continue
                        
                    # Get room name
                    room_id = tile.get('room_id', -1)
                    room_name = "Unknown"
                    room_row = room_df[room_df['room_id'] == room_id]
                    if not room_row.empty:
                        room_name = room_row['room_name'].values[0]
                    
                    # Get classification and cut side from tile
                    classification = tile.get('classification', 'unknown')
                    cut_side = tile.get('cut_side', None)
                    
                    # Get measurements
                    polygon = tile['polygon']
                    minx, miny, maxx, maxy = polygon.bounds
                    measured_width = maxx - minx
                    measured_height = maxy - miny
                    
                    # For split tiles, include original reference if available
                    is_split = tile.get('is_split', False)
                    part_index = tile.get('part_index', None)
                    original_tile_index = tile.get('original_tile_index', None)
                    
                    all_tiles.append({
                        'apartment_name': apt_name,
                        'room_id': room_id,
                        'room_name': room_name,
                        'tile_index': tile_idx,
                        'polygon': polygon,
                        'orientation': apt_data['orientation'],
                        'measured_width': measured_width,
                        'measured_height': measured_height,
                        'classification': classification,
                        'cut_side': cut_side,
                        'is_split': is_split,
                        'part_index': part_index,
                        'original_tile_index': original_tile_index
                    })
            
            # Convert to DataFrame
            tiles_df = pd.DataFrame(all_tiles)
            
            # Debug tiles_df before small cuts identification
            print(f"tiles_df created with {len(tiles_df)} tiles")
            print(f"Classification counts: {tiles_df['classification'].value_counts().to_dict()}")
            
            # Identify small cut tiles - use the exact function from Colab
            small_tiles_df, small_tile_count = identify_small_cut_tiles(
                tiles_df, room_df, has_pattern, size_threshold)
            
            print(f"Small cuts identified: {small_tile_count}")
            
            # Generate visualization
            small_tiles_plot_b64 = visualize_small_tiles(
                tiles_df, small_tiles_df, room_df, size_threshold)
            
            # If exclude_small_cuts is True, update the apartments_data
            if exclude_small_cuts and small_tile_count > 0:
                for apt_name, apt_data in apartments_data.items():
                    updated_tiles = []
                    for tile_idx, tile in enumerate(apt_data['tiles']):
                        # Check if this tile is in small_tiles_df
                        is_small_cut = False
                        if not small_tiles_df.empty:
                            matching_small_tiles = small_tiles_df[
                                (small_tiles_df['apartment_name'] == apt_name) & 
                                (small_tiles_df['tile_index'] == tile_idx)
                            ]
                            if not matching_small_tiles.empty:
                                is_small_cut = True
                        
                        # Keep tile if it's not a small cut
                        if not is_small_cut:
                            updated_tiles.append(tile)
                    
                    # Update the tiles list
                    apt_data['tiles'] = updated_tiles
                
                print(f"Excluded {small_tile_count} small tiles from apartments_data")
            
            # Store small cuts data in session
            session['small_cuts_data'] = {
                'size_threshold': size_threshold,
                'exclude_small_cuts': exclude_small_cuts,
                'small_tile_count': small_tile_count,
                'small_tiles_indices': small_tiles_df['tile_index'].tolist() if not small_tiles_df.empty else [],
                'updated_apartments_data': serialize_apartments_data(apartments_data) if exclude_small_cuts else None
            }
            
            # Prepare the location summary
            location_summary = []
            if not small_tiles_df.empty:
                location_data = small_tiles_df.groupby(['apartment_name', 'room_name']).size().reset_index(name='count')
                for _, row in location_data.iterrows():
                    location_summary.append({
                        'apartment': row['apartment_name'],
                        'room': row['room_name'],
                        'count': int(row['count'])
                    })
            
            # Prepare size distribution
            size_distribution = []
            if not small_tiles_df.empty and 'cut_side' in small_tiles_df.columns:
                # Create bins for the cut dimensions
                bins = [0, 2, 4, 6, 8, 10]
                labels = ['0-2mm', '2-4mm', '4-6mm', '6-8mm', '8-10mm']
                
                # Apply binning
                small_tiles_df['size_range'] = pd.cut(
                    small_tiles_df['cut_side'], 
                    bins=bins, 
                    labels=labels, 
                    include_lowest=True)
                
                # Count by size range
                size_counts = small_tiles_df['size_range'].value_counts().reset_index()
                size_counts.columns = ['range', 'count']
                
                # Convert to list of dicts
                for _, row in size_counts.iterrows():
                    size_distribution.append({
                        'range': row['range'],
                        'count': int(row['count'])
                    })
            
            # Prepare small tiles list (limit to 20 for display)
            small_tiles_list = []
            if not small_tiles_df.empty:
                for _, tile in small_tiles_df.head(20).iterrows():
                    small_tiles_list.append({
                        'apartment': tile['apartment_name'],
                        'room': tile['room_name'],
                        'classification': tile['classification'],
                        'cut_dimension': float(tile['cut_side']) if 'cut_side' in tile else 0
                    })
            
            return jsonify({
                'success': True,
                'small_tiles_plot': small_tiles_plot_b64,
                'small_tile_count': small_tile_count,
                'total_cut_count': len(tiles_df[tiles_df['classification'].isin(['cut_x', 'cut_y', 'all_cut'])]),
                'location_summary': location_summary,
                'size_distribution': size_distribution,
                'small_tiles_list': small_tiles_list
            })
        
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            print(traceback_str)
            return jsonify({
                'error': f'Error identifying small cuts: {str(e)}',
                'traceback': traceback_str
            })
    
    # GET request - render the template
    try:
        # Check if required data exists
        if 'classification_data' not in session:
            return redirect(url_for('step5'))
        
        return render_template('step6.html')
    
    except Exception as e:
        print(f"Error in step6 GET: {str(e)}")
        return redirect(url_for('step5'))

def identify_small_cut_tiles(tiles_df, final_room_df, has_pattern=False, size_threshold=10):
    """
    Identify cut tiles with dimension less than the specified threshold
    """
    print(f"\n🔍 Identifying cut tiles with dimension < {size_threshold}mm...")
    
    # Find small cut tiles based on pattern mode
    if has_pattern:
        # With pattern: check cut_x and cut_y separately
        small_x_tiles = tiles_df[
            (tiles_df['classification'] == 'cut_x') & 
            (tiles_df['cut_side'] < size_threshold)
        ].copy()
        
        small_y_tiles = tiles_df[
            (tiles_df['classification'] == 'cut_y') & 
            (tiles_df['cut_side'] < size_threshold)
        ].copy()
        
        # Combine small x and y tiles
        small_tiles_df = pd.concat([small_x_tiles, small_y_tiles])
        
    else:
        # No pattern: check all_cut tiles
        small_tiles_df = tiles_df[
            (tiles_df['classification'] == 'all_cut') & 
            (tiles_df['cut_side'] < size_threshold)
        ].copy()
    
    # Count small tiles
    small_tile_count = len(small_tiles_df)
    total_cut_count = len(tiles_df[tiles_df['classification'].isin(['cut_x', 'cut_y', 'all_cut'])])
    
    if small_tile_count > 0:
        print(f"✅ Found {small_tile_count} cut tiles smaller than {size_threshold}mm")
        print(f"   This represents {small_tile_count/total_cut_count*100:.1f}% of all cut tiles")
        
        # Group by apartment and room
        location_summary = small_tiles_df.groupby(['apartment_name', 'room_name']).size().reset_index(name='count')
        print("\n📊 Small Tile Distribution by Location:")
        for _, row in location_summary.iterrows():
            print(f"   {row['apartment_name']} - {row['room_name']}: {row['count']} small tiles")
    else:
        print(f"✅ No cut tiles smaller than {size_threshold}mm found")
    
    return small_tiles_df, small_tile_count

def visualize_small_tiles(tiles_df, small_tiles_df, final_room_df, size_threshold=10):
    """Visualize small cut tiles in context of all tiles"""
    print(f"\n🎨 Visualizing tiles with cut dimension < {size_threshold}mm...")
    
    plt.figure(figsize=(16, 16))
    
    # Plot room outlines
    for _, room in final_room_df.iterrows():
        room_poly = room['polygon']
        if hasattr(room_poly, 'exterior'):
            x, y = room_poly.exterior.xy
            plt.plot(x, y, color='black', linewidth=1.5, alpha=0.8)
            
            # Add room label if centroid coordinates are available
            if 'centroid_x' in room and 'centroid_y' in room:
                plt.text(room['centroid_x'], room['centroid_y'], 
                        f"{room['apartment_name']}-{room['room_name']}", 
                        fontsize=10, ha='center', va='center', 
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # Define colors for different types
    colors = {
        'full': 'lightgray',       # Light gray for full tiles
        'irregular': 'lightblue',  # Light blue for irregular tiles
        'cut_normal': 'lightgreen', # Light green for normal cut tiles
        'small_cut': 'red',        # Red for small cut tiles
        'unknown': 'gray'          # Gray for unknown types
    }
    
    # First, plot all non-small tiles with lighter colors
    for _, tile in tiles_df.iterrows():
        classification = tile['classification']
        
        # Check if this is a cut tile
        is_cut = classification in ['cut_x', 'cut_y', 'all_cut']
        
        # For cut tiles, check if they're in the small tiles DataFrame
        is_small_cut = False
        if is_cut and not small_tiles_df.empty:
            matching_tiles = small_tiles_df[
                (small_tiles_df['tile_index'] == tile['tile_index']) & 
                (small_tiles_df['apartment_name'] == tile['apartment_name'])
            ]
            is_small_cut = len(matching_tiles) > 0
        
        # Skip small cut tiles for now (will plot them later)
        if is_small_cut:
            continue
            
        # Choose color based on classification
        if classification == 'full':
            color = colors['full']
        elif classification == 'irregular':
            color = colors['irregular']
        elif is_cut:
            color = colors['cut_normal']
        else:
            color = colors['unknown']
            
        poly = tile['polygon']
        
        try:
            if isinstance(poly, Polygon) and poly.is_valid and not poly.is_empty:
                x, y = poly.exterior.xy
                plt.fill(x, y, color=color, alpha=0.5, edgecolor='black', linewidth=0.5)
            elif isinstance(poly, MultiPolygon):
                for part in poly.geoms:
                    if part.is_valid and not part.is_empty:
                        x, y = part.exterior.xy
                        plt.fill(x, y, color=color, alpha=0.5, edgecolor='black', linewidth=0.5)
        except Exception as e:
            print(f"Error plotting tile: {e}")
    
    # Now plot the small cut tiles on top with bright red color
    for _, tile in small_tiles_df.iterrows():
        poly = tile['polygon']
        
        try:
            if isinstance(poly, Polygon) and poly.is_valid and not poly.is_empty:
                x, y = poly.exterior.xy
                plt.fill(x, y, color=colors['small_cut'], alpha=0.9, edgecolor='black', linewidth=1.0)
                
                # Add cut dimension label
                if hasattr(poly, 'centroid'):
                    centroid = poly.centroid
                    cut_size = tile['cut_side']
                    plt.text(centroid.x, centroid.y, f"{cut_size:.1f}", 
                            fontsize=8, ha='center', va='center', color='white',
                            bbox=dict(facecolor='red', alpha=0.7, boxstyle='round,pad=0.1'))
                
            elif isinstance(poly, MultiPolygon):
                for part in poly.geoms:
                    if part.is_valid and not part.is_empty:
                        x, y = part.exterior.xy
                        plt.fill(x, y, color=colors['small_cut'], alpha=0.9, edgecolor='black', linewidth=1.0)
        except Exception as e:
            print(f"Error plotting small tile: {e}")
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['full'], alpha=0.5, edgecolor='black', label='Full Tiles'),
        Patch(facecolor=colors['irregular'], alpha=0.5, edgecolor='black', label='Irregular Tiles'),
        Patch(facecolor=colors['cut_normal'], alpha=0.5, edgecolor='black', label='Normal Cut Tiles'),
        Patch(facecolor=colors['small_cut'], alpha=0.9, edgecolor='black', label=f'Small Cut Tiles (< {size_threshold}mm)')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(f"Small Cut Tiles (< {size_threshold}mm) Visualization")
    plt.axis('equal')
    plt.grid(False)
    plt.tight_layout()
    
    # Save figure to buffer and return base64 encoded string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/step7', methods=['GET', 'POST'])
def step7():
    """Step 7: Export Remaining Tiles with Wastage Analysis"""
    if request.method == 'POST':
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Invalid data format'})
        
        try:
            # Get data from session
            classification_data = session.get('classification_data', {})
            small_cuts_data = session.get('small_cuts_data', {})
            
            # Get apartments data based on whether small cuts were excluded
            if small_cuts_data.get('exclude_small_cuts', False) and small_cuts_data.get('updated_apartments_data'):
                apartments_data = deserialize_apartments_data(small_cuts_data.get('updated_apartments_data', {}))
            else:
                apartments_data = deserialize_apartments_data(classification_data.get('classified_apartments_data', {}))
            
            has_pattern = classification_data.get('has_pattern', False)
            
            # Get room_df for room information
            room_df = pd.DataFrame(session.get('room_df', []))
            room_polygons = deserialize_rooms(session.get('room_polygons', []))
            
            # Add polygons back to room_df for processing
            if len(room_polygons) == len(room_df):
                room_df['polygon'] = room_polygons
            
            # Get export settings
            export_prefix = data.get('export_prefix', 'final_tiles_export')
            export_format = data.get('export_format', 'excel')
            project_name = data.get('project_name', 'Project_1')
            include_visualization = data.get('include_visualization', True)
            
            # Generate filename
            filename = f"{export_prefix}_{project_name}.xlsx"
            
            # Get size threshold
            size_threshold = small_cuts_data.get('size_threshold', 10)
            
            # Prepare export data according to pattern mode
            export_info = {}
            
            if has_pattern:
                # --- Process cut_x tiles ---
                cut_x_data = []
                for apt_name, apt_data in apartments_data.items():
                    cut_x_tiles_list = [t for t in apt_data['tiles'] if t.get('classification') == 'cut_x']
                    
                    for tile in cut_x_tiles_list:
                        cut_side = tile.get('cut_side', 0)
                        room_id = tile.get('room_id', -1)
                        
                        # Get room name
                        room_name = "Unknown"
                        room_rows = room_df[room_df['room_id'] == room_id]
                        if not room_rows.empty:
                            room_name = room_rows.iloc[0]['room_name']
                        
                        cut_x_data.append({
                            'APPARTMENT NUMBER': apt_name,
                            'CUT SIDE (mm)': round(cut_side),
                            'LOCATION': room_name,
                            'COUNT': 1
                        })
                
                # Group by apartment, room and cut side for cut_x_simple
                cut_x_simple = pd.DataFrame(cut_x_data) if cut_x_data else pd.DataFrame()
                if not cut_x_simple.empty:
                    cut_x_simple = cut_x_simple.groupby(['APPARTMENT NUMBER', 'CUT SIDE (mm)', 'LOCATION']).sum().reset_index()
                    cut_x_simple = cut_x_simple.sort_values(['APPARTMENT NUMBER', 'CUT SIDE (mm)'])
                
                # --- Process cut_y tiles ---
                cut_y_data = []
                for apt_name, apt_data in apartments_data.items():
                    cut_y_tiles_list = [t for t in apt_data['tiles'] if t.get('classification') == 'cut_y']
                    
                    for tile in cut_y_tiles_list:
                        cut_side = tile.get('cut_side', 0)
                        room_id = tile.get('room_id', -1)
                        
                        # Get room name
                        room_name = "Unknown"
                        room_rows = room_df[room_df['room_id'] == room_id]
                        if not room_rows.empty:
                            room_name = room_rows.iloc[0]['room_name']
                        
                        cut_y_data.append({
                            'APPARTMENT NUMBER': apt_name,
                            'CUT SIDE (mm)': round(cut_side),
                            'LOCATION': room_name,
                            'COUNT': 1
                        })
                
                # Group by apartment, room and cut side for cut_y_simple
                cut_y_simple = pd.DataFrame(cut_y_data) if cut_y_data else pd.DataFrame()
                if not cut_y_simple.empty:
                    cut_y_simple = cut_y_simple.groupby(['APPARTMENT NUMBER', 'CUT SIDE (mm)', 'LOCATION']).sum().reset_index()
                    cut_y_simple = cut_y_simple.sort_values(['APPARTMENT NUMBER', 'CUT SIDE (mm)'])
                
                # Store in export_info - CRITICAL for Step 8
                export_info = {
                    'cut_x_simple': cut_x_simple,
                    'cut_y_simple': cut_y_simple,
                    'has_pattern': has_pattern,
                    'tile_width': 600,  # Default width, update if available
                    'tile_height': 600  # Default height, update if available
                }
                
                # Update with actual tile dimensions if available
                for apt_name, apt_data in apartments_data.items():
                    if apt_data['tiles']:
                        first_tile = apt_data['tiles'][0]
                        if 'actual_width' in first_tile and 'actual_height' in first_tile:
                            export_info['tile_width'] = first_tile['actual_width']
                            export_info['tile_height'] = first_tile['actual_height']
                            break
            else:
                # --- Process all_cut tiles ---
                all_cut_data = []
                for apt_name, apt_data in apartments_data.items():
                    all_cut_tiles_list = [t for t in apt_data['tiles'] if t.get('classification') == 'all_cut']
                    
                    for tile in all_cut_tiles_list:
                        cut_side = tile.get('cut_side', 0)
                        room_id = tile.get('room_id', -1)
                        
                        # Get room name
                        room_name = "Unknown"
                        room_rows = room_df[room_df['room_id'] == room_id]
                        if not room_rows.empty:
                            room_name = room_rows.iloc[0]['room_name']
                        
                        all_cut_data.append({
                            'APPARTMENT NUMBER': apt_name,
                            'CUT DIMENSION (mm)': round(cut_side),
                            'LOCATION': room_name,
                            'COUNT': 1
                        })
                
                # Group by apartment, room and cut dimension
                all_cut_simple = pd.DataFrame(all_cut_data) if all_cut_data else pd.DataFrame()
                if not all_cut_simple.empty:
                    all_cut_simple = all_cut_simple.groupby(['APPARTMENT NUMBER', 'CUT DIMENSION (mm)', 'LOCATION']).sum().reset_index()
                    all_cut_simple = all_cut_simple.sort_values(['APPARTMENT NUMBER', 'CUT DIMENSION (mm)'])
                
                # Store in export_info - CRITICAL for Step 8
                export_info = {
                    'all_cut_simple': all_cut_simple,
                    'has_pattern': has_pattern,
                    'tile_width': 600,  # Default width, update if available
                    'tile_height': 600  # Default height, update if available
                }
                
                # Update with actual tile dimensions if available
                for apt_name, apt_data in apartments_data.items():
                    if apt_data['tiles']:
                        first_tile = apt_data['tiles'][0]
                        if 'actual_width' in first_tile and 'actual_height' in first_tile:
                            export_info['tile_width'] = first_tile['actual_width']
                            export_info['tile_height'] = first_tile['actual_height']
                            break
            
            # Calculate tile counts and wastage
            full_tiles = classification_data.get('full_tiles', 0)
            irregular_tiles = classification_data.get('irregular_tiles', 0)
            small_tiles = small_cuts_data.get('small_tile_count', 0)
            
            # Calculate actual wastage by apartment
            wastage = []
            total_area = 0
            total_wastage = 0
            
            for apt_name, apt_data in apartments_data.items():
                # Calculate actual area
                apt_tiles = len(apt_data['tiles'])
                apt_area = apt_tiles * 0.36  # Assuming 600x600mm tiles
                
                # Use a formula based on cut tiles rather than random
                cut_tiles = sum(1 for t in apt_data['tiles'] 
                             if t.get('classification') in ['cut_x', 'cut_y', 'all_cut'])
                apt_wastage_pct = (cut_tiles / apt_tiles * 10) if apt_tiles > 0 else 0
                apt_wastage = apt_area * apt_wastage_pct / 100
                
                wastage.append({
                    'apartment': apt_name,
                    'area': round(apt_area, 2),
                    'percentage': round(apt_wastage_pct, 2)
                })
                
                total_area += apt_area
                total_wastage += apt_wastage
            
            # Calculate overall wastage percentage
            overall_wastage_pct = (total_wastage / total_area) * 100 if total_area > 0 else 0
            
            # Store export data in session
            session['export_data'] = {
                'file_name': filename,
                'export_format': export_format,
                'project_name': project_name,
                'apartments': [apt for apt in apartments_data.keys()],
                'total_tiles': classification_data.get('total_tiles', 0),
                'wastage_percentage': overall_wastage_pct
            }
            
            # Make DataFrames serializable for session
            serialized_export_info = make_serializable_for_session(export_info)
            
            # Store export_info in session for step 8 - THIS IS CRITICAL
            session['export_info'] = serialized_export_info
            
            # Prepare summary data for UI
            summary = [
                {'type': 'Full Tiles', 'count': full_tiles},
                {'type': 'Irregular Tiles', 'count': irregular_tiles}
            ]
            
            if has_pattern:
                cut_x_tiles = classification_data.get('cut_x_tiles', 0)
                cut_y_tiles = classification_data.get('cut_y_tiles', 0)
                summary.append({'type': 'Cut Tiles (X)', 'count': cut_x_tiles})
                summary.append({'type': 'Cut Tiles (Y)', 'count': cut_y_tiles})
            else:
                all_cut_tiles = classification_data.get('all_cut_tiles', 0)
                summary.append({'type': 'All Cut Tiles', 'count': all_cut_tiles})
            
            if small_tiles > 0:
                summary.append({'type': 'Small Cut Tiles (excluded)', 'count': small_tiles})
            
            summary.append({'type': 'Total Tiles', 'count': classification_data.get('total_tiles', 0)})
            
            return jsonify({
                'success': True,
                'results': {
                    'file_name': filename,
                    'wastage': wastage,
                    'summary': summary
                }
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error generating report: {str(e)}'})
    
    # GET request - render the template
    try:
        # Check if required data exists
        if 'small_cuts_data' not in session:
            return redirect(url_for('step6'))
        
        return render_template('step7.html')
    
    except Exception as e:
        print(f"Error in step7 GET: {str(e)}")
        return redirect(url_for('step6'))
    

def make_serializable_for_session(export_info):
    """Convert DataFrames to dictionaries for session storage"""
    result = {}
    
    # Copy everything except DataFrames
    for key, value in export_info.items():
        if isinstance(value, pd.DataFrame):
            # Convert DataFrame to records
            result[key] = value.to_dict('records')
        else:
            result[key] = value
    
    return result

def generate_optimization_report(optimization_results, inventory_data=None):
    """
    Generate a detailed Excel report of optimization results
    
    Parameters:
    -----------
    optimization_results : dict
        Optimization results from optimize_cut_pieces function
    inventory_data : dict, optional
        Inventory data if available
        
    Returns:
    --------
    BytesIO
        Excel file as a BytesIO object
    """
    import pandas as pd
    import io
    from datetime import datetime
    
    # Create an in-memory output file
    output = io.BytesIO()
    
    # Convert matches to DataFrame
    matches_df = pd.DataFrame(optimization_results['simplified_matches'])
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Create summary sheet
        summary_data = {
            'Metric': [
                'Total Matches',
                'Within-Apartment Matches',
                'Cross-Apartment Matches',
                'Total Material Saved (mm²)',
                'Total Material Saved (m²)',
                'Optimization Date'
            ],
            'Value': [
                optimization_results['matched_count'],
                optimization_results['within_apartment_matches'],
                optimization_results['cross_apartment_matches'],
                optimization_results['total_savings'],
                optimization_results['total_savings'] / 1000000,
                timestamp
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Format the summary sheet
        summary_sheet = writer.sheets['Summary']
        summary_sheet.set_column('A:A', 25)
        summary_sheet.set_column('B:B', 20)
        
        # Create matches sheet
        if not matches_df.empty:
            matches_df.to_excel(writer, sheet_name='Matches', index=False)
            
            # Format the matches sheet
            matches_sheet = writer.sheets['Matches']
            matches_sheet.set_column('A:E', 15)
            
            # Add conditional formatting for cross-apartment matches
            matches_sheet.conditional_format('A2:E1000', {
                'type': 'formula',
                'criteria': '=$E2="True"',
                'format': writer.book.add_format({'bg_color': '#ffffcc'})
            })
        
        # Create inventory sheet if inventory data is available
        if inventory_data:
            has_pattern = inventory_data.get('has_pattern', True)
            
            if has_pattern:
                # For pattern-based inventory
                x_inventory = inventory_data.get('cut_x_inventory', pd.DataFrame())
                y_inventory = inventory_data.get('cut_y_inventory', pd.DataFrame())
                
                if not x_inventory.empty:
                    x_inventory.to_excel(writer, sheet_name='X Inventory', index=False)
                    x_sheet = writer.sheets['X Inventory']
                    x_sheet.set_column('A:C', 15)
                
                if not y_inventory.empty:
                    y_inventory.to_excel(writer, sheet_name='Y Inventory', index=False)
                    y_sheet = writer.sheets['Y Inventory']
                    y_sheet.set_column('A:C', 15)
            else:
                # For non-pattern inventory
                all_inventory = inventory_data.get('all_cut_inventory', pd.DataFrame())
                
                if not all_inventory.empty:
                    all_inventory.to_excel(writer, sheet_name='All Inventory', index=False)
                    all_sheet = writer.sheets['All Inventory']
                    all_sheet.set_column('A:C', 15)
        
        # Create instructions sheet
        instructions_data = {
            'Instructions': [
                'This report contains the results of tile cut piece optimization.',
                '',
                'Sheets in this workbook:',
                '1. Summary - Overall optimization statistics',
                '2. Matches - Detailed list of all matched cut pieces',
                '3. X/Y/All Inventory - Inventory data used for optimization (if available)',
                '',
                'Match Types:',
                '- Within-Apartment: Matches between pieces in the same apartment',
                '- Cross-Apartment: Matches between pieces in different apartments (highlighted in yellow)',
                '',
                'To implement these optimizations in your project:',
                '1. Use the matched pieces as indicated in the report',
                '2. Verify all matches before cutting to ensure they meet your requirements',
                '3. Update your inventory with any remaining unmatched pieces for future projects'
            ]
        }
        
        instructions_df = pd.DataFrame(instructions_data)
        instructions_df.to_excel(writer, sheet_name='Instructions', index=False)
        
        # Format the instructions sheet
        instructions_sheet = writer.sheets['Instructions']
        instructions_sheet.set_column('A:A', 80)
    
    # Reset the buffer position
    output.seek(0)
    
    return output

# Add routes for inventory template download and optimization report download
@app.route('/create_inventory_template_route')
def create_inventory_template_route():
    """Create inventory template file for download"""
    try:
        # Get pattern mode from session
        classification_data = session.get('classification_data', {})
        has_pattern = classification_data.get('has_pattern', True)
        
        # Create optimizer instance
        optimizer = OptimizationProcessor()
        
        # Create template
        output = optimizer.create_inventory_template(has_pattern)
        
        # Create response
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response.headers['Content-Disposition'] = 'attachment; filename=inventory_template.xlsx'
        
        return response
        
    except Exception as e:
        return jsonify({'error': f'Error creating template: {str(e)}'})

@app.route('/step8/upload_inventory', methods=['POST'])
def upload_inventory():
    """Upload and process inventory file"""
    try:
        # Get uploaded file
        if 'inventory_file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['inventory_file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        # Create optimization processor
        optimizer = OptimizationProcessor()
        
        # Process inventory file
        inventory_data = optimizer.load_and_process_inventory(file)
        
        if 'error' in inventory_data:
            return jsonify({'error': inventory_data['error']})
        
        # Debug inventory data
        print("\nInventory data loaded successfully:")
        for key, value in inventory_data.items():
            if key != 'has_pattern':
                if isinstance(value, pd.DataFrame):
                    print(f"  {key}: DataFrame with {len(value)} rows")
                    if not value.empty:
                        print(f"  {key} columns: {list(value.columns)}")
                        print(f"  {key} first row: {value.iloc[0].to_dict()}")
                else:
                    print(f"  {key}: {value}")
        
        # Store inventory data in session
        session['inventory_data'] = inventory_data
        
        # Count inventory pieces
        inventory_count = 0
        if inventory_data.get('has_pattern', True):
            if 'cut_x_inventory' in inventory_data and isinstance(inventory_data['cut_x_inventory'], pd.DataFrame):
                inventory_count += inventory_data['cut_x_inventory']['Count'].sum() if 'Count' in inventory_data['cut_x_inventory'].columns else len(inventory_data['cut_x_inventory'])
            if 'cut_y_inventory' in inventory_data and isinstance(inventory_data['cut_y_inventory'], pd.DataFrame):
                inventory_count += inventory_data['cut_y_inventory']['Count'].sum() if 'Count' in inventory_data['cut_y_inventory'].columns else len(inventory_data['cut_y_inventory'])
        else:
            if 'all_cut_inventory' in inventory_data and isinstance(inventory_data['all_cut_inventory'], pd.DataFrame):
                inventory_count += inventory_data['all_cut_inventory']['Count'].sum() if 'Count' in inventory_data['all_cut_inventory'].columns else len(inventory_data['all_cut_inventory'])
        
        print(f"Total inventory count: {inventory_count}")
        
        # Make session data serializable
        serializable_inventory = {}
        for key, value in inventory_data.items():
            if isinstance(value, pd.DataFrame):
                serializable_inventory[key] = value.to_dict('records')
            else:
                serializable_inventory[key] = value
        
        session['inventory_data'] = serializable_inventory
        
        return jsonify({
            'success': True,
            'message': 'Inventory file uploaded successfully',
            'inventory_count': int(inventory_count)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error uploading inventory: {str(e)}'})

@app.route('/step8/generate_report', methods=['GET'])
def generate_optimization_report():
    try:
        # Create OptimizationProcessor instance
        optimizer = OptimizationProcessor()
        
        # Get optimization results from session
        optimization_results = session.get('optimization_results', {})
        inventory_data = session.get('inventory_data', None)
        
        if not optimization_results:
            return jsonify({'error': 'No optimization results found. Please run optimization first.'})
        
        # Generate report
        output = optimizer.generate_optimization_report(optimization_results, inventory_data)
        
        # Create response
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response.headers['Content-Disposition'] = 'attachment; filename=optimization_report.xlsx'
        
        return response
        
    except Exception as e:
        return jsonify({'error': f'Error generating report: {str(e)}'})

def create_inventory_template(has_pattern=True):
    """Create an inventory template file for users to fill in"""
    import pandas as pd
    import io
    
    # Create an in-memory output file
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        if has_pattern:
            # Create Cut X template with simplified columns
            cut_x_template = pd.DataFrame(columns=[
                'Remaining Size (mm)', 'Location', 'Count'
            ])
            cut_x_template.to_excel(writer, sheet_name='Cut X', index=False)
            
            # Create Cut Y template with simplified columns
            cut_y_template = pd.DataFrame(columns=[
                'Remaining Size (mm)', 'Location', 'Count'
            ])
            cut_y_template.to_excel(writer, sheet_name='Cut Y', index=False)
            
            # Add a sample data sheet
            sample_data = pd.DataFrame([
                {'Remaining Size (mm)': 150, 'Location': 'INV', 'Count': 2},
                {'Remaining Size (mm)': 220, 'Location': 'INV', 'Count': 3},
                {'Remaining Size (mm)': 350, 'Location': 'INV', 'Count': 1},
                {'Remaining Size (mm)': 480, 'Location': 'INV', 'Count': 4}
            ])
            sample_data.to_excel(writer, sheet_name='Sample Data', index=False)
        else:
            # Create All Cut template with simplified columns
            all_cut_template = pd.DataFrame(columns=[
                'Remaining Size (mm)', 'Location', 'Count'
            ])
            all_cut_template.to_excel(writer, sheet_name='All Cut', index=False)
            
            # Add a sample data sheet
            sample_data = pd.DataFrame([
                {'Remaining Size (mm)': 150, 'Location': 'INV', 'Count': 2},
                {'Remaining Size (mm)': 220, 'Location': 'INV', 'Count': 3},
                {'Remaining Size (mm)': 350, 'Location': 'INV', 'Count': 1},
                {'Remaining Size (mm)': 480, 'Location': 'INV', 'Count': 4}
            ])
            sample_data.to_excel(writer, sheet_name='Sample Data', index=False)
        
        # Add instructions sheet
        instructions = pd.DataFrame([
            {'Instructions': 'Fill in the sheets with your inventory data for remaining pieces'},
            {'Instructions': 'Remaining Size (mm): The size of the remaining piece in millimeters'},
            {'Instructions': 'Location: Storage location (default is "INV" for inventory)'},
            {'Instructions': 'Count: Number of pieces with these specifications'},
            {'Instructions': ''},
            {'Instructions': 'These are remaining pieces from previous jobs that can be used in future projects.'},
            {'Instructions': 'No need to calculate cut sizes as these are just the remaining pieces in inventory.'}
        ])
        instructions.to_excel(writer, sheet_name='Instructions', index=False)
    
    # Reset the buffer position
    output.seek(0)
    
    return output

def load_and_process_inventory(file):
    """
    Load and process an inventory file
    
    Parameters:
    -----------
    file : FileStorage
        The uploaded inventory file
        
    Returns:
    --------
    dict
        Processed inventory data
    """
    import pandas as pd
    
    try:
        # Check file type
        if file.filename.endswith('.xlsx'):
            # Read Excel file
            xl = pd.ExcelFile(file)
            
            # Check if required sheets exist
            if 'Instructions' in xl.sheet_names:
                # Remove Instructions sheet for processing
                sheet_names = [s for s in xl.sheet_names if s != 'Instructions']
            else:
                sheet_names = xl.sheet_names
            
            # Check for pattern-based inventory
            has_pattern = 'Cut X' in sheet_names and 'Cut Y' in sheet_names
            all_cut = 'All Cut' in sheet_names
            
            if not has_pattern and not all_cut:
                return {'error': 'Invalid inventory file format. Expected sheets "Cut X" and "Cut Y" or "All Cut".'}
            
            inventory_data = {'has_pattern': has_pattern}
            
            if has_pattern:
                # Read Cut X inventory
                if 'Cut X' in sheet_names:
                    cut_x_df = pd.read_excel(file, sheet_name='Cut X')
                    if not all(col in cut_x_df.columns for col in ['Remaining Size (mm)', 'Location', 'Count']):
                        return {'error': 'Cut X sheet missing required columns: "Remaining Size (mm)", "Location", "Count"'}
                    inventory_data['cut_x_inventory'] = cut_x_df
                
                # Read Cut Y inventory
                if 'Cut Y' in sheet_names:
                    cut_y_df = pd.read_excel(file, sheet_name='Cut Y')
                    if not all(col in cut_y_df.columns for col in ['Remaining Size (mm)', 'Location', 'Count']):
                        return {'error': 'Cut Y sheet missing required columns: "Remaining Size (mm)", "Location", "Count"'}
                    inventory_data['cut_y_inventory'] = cut_y_df
            else:
                # Read All Cut inventory
                all_cut_df = pd.read_excel(file, sheet_name='All Cut')
                if not all(col in all_cut_df.columns for col in ['Remaining Size (mm)', 'Location', 'Count']):
                    return {'error': 'All Cut sheet missing required columns: "Remaining Size (mm)", "Location", "Count"'}
                inventory_data['all_cut_inventory'] = all_cut_df
            
            return inventory_data
        else:
            return {'error': 'Unsupported file format. Please upload an Excel (.xlsx) file.'}
    
    except Exception as e:
        return {'error': f'Error processing inventory file: {str(e)}'}

@app.route('/step8', methods=['GET', 'POST'])
def step8():
    """Step 8: Cut Piece Optimization (Beta)"""
    if request.method == 'GET':
        try:
            # Check if export_info exists in session
            if 'export_info' not in session:
                # Use flash instead of failing silently
                flash('Please complete the export step first.', 'warning')
                return redirect(url_for('step7'))
            
            # Get apartments for the selection UI
            apartments = []
            room_df = pd.DataFrame(session.get('room_df', []))
            if not room_df.empty and 'apartment_name' in room_df.columns:
                apartments = sorted(room_df['apartment_name'].unique())
            
            # Get pattern mode from classification data
            classification_data = session.get('classification_data', {})
            has_pattern = classification_data.get('has_pattern', True)
            
            return render_template('step8.html', 
                                  apartments=apartments,
                                  has_pattern=has_pattern)
        except Exception as e:
            print(f"Error in step8 GET: {str(e)}")
            import traceback
            traceback.print_exc()
            # Use flash instead of failing silently
            flash('An error occurred. Please try again.', 'error')
            return redirect(url_for('step7'))
    else:
        # POST request handling
        return jsonify({'error': 'Direct POST to /step8 is not supported. Please use /step8/full_process instead.'})

def _filter_data_by_apartments(self, cut_data, selected_apartments):
    """
    Filter cut data to include only selected apartments
    
    Parameters:
    - cut_data: Dict with cut data
    - selected_apartments: List of apartments to include
    
    Returns:
    - Dict with filtered cut data
    """
    # Get pattern mode
    has_pattern = cut_data.get('has_pattern', True)
    
    # Create copy of cut_data
    filtered_data = {
        'has_pattern': has_pattern,
        'tile_width': cut_data.get('tile_width', 600),
        'tile_height': cut_data.get('tile_height', 600)
    }
    
    # Filter the data
    if has_pattern:
        # Filter X data
        cut_x_data = cut_data.get('cut_x_data', pd.DataFrame())
        if not cut_x_data.empty:
            filtered_data['cut_x_data'] = cut_x_data[cut_x_data['Apartment'].isin(selected_apartments)].copy()
        
        # Filter Y data
        cut_y_data = cut_data.get('cut_y_data', pd.DataFrame())
        if not cut_y_data.empty:
            filtered_data['cut_y_data'] = cut_y_data[cut_y_data['Apartment'].isin(selected_apartments)].copy()
    else:
        # Filter All data
        all_cut_data = cut_data.get('all_cut_data', pd.DataFrame())
        if not all_cut_data.empty:
            filtered_data['all_cut_data'] = all_cut_data[all_cut_data['Apartment'].isin(selected_apartments)].copy()
    
    return filtered_data


@app.route('/download/<filename>')
def download_file(filename):
    """Download exported file"""
    import io
    import pandas as pd
    from flask import send_file
    
    if 'export_data' not in session:
        return "No export data available. Please complete the export step first."
    
    try:
        # Get export data and classification data from session
        export_data = session.get('export_data', {})
        classification_data = session.get('classification_data', {})
        small_cuts_data = session.get('small_cuts_data', {})
        apartments_data = deserialize_apartments_data(classification_data.get('classified_apartments_data', {}))
        has_pattern = classification_data.get('has_pattern', False)
        
        # Get basic stats
        total_tiles = export_data.get('total_tiles', 0)
        full_tiles = classification_data.get('full_tiles', 0)
        irregular_tiles = classification_data.get('irregular_tiles', 0)
        small_tiles_count = small_cuts_data.get('small_tile_count', 0)
        
        if has_pattern:
            cut_x_tiles = classification_data.get('cut_x_tiles', 0)
            cut_y_tiles = classification_data.get('cut_y_tiles', 0)
        else:
            all_cut_tiles = classification_data.get('all_cut_tiles', 0)
        
        # Create an in-memory output file
        output = io.BytesIO()
        
        # Create a Excel writer using the in-memory output file
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Create wastage analysis sheet
            wastage_data = []
            apartments = export_data.get('apartments', [])
            
            for apt_name in apartments:
                # Generate some example data for each apartment
                apt_tiles = [t for apt, tiles in apartments_data.items() 
                            for t in tiles['tiles'] if apt == apt_name]
                
                # Calculate area (in real implementation, would use actual areas)
                apt_area = len(apt_tiles) * 0.36  # Assuming 600x600mm tiles (0.36 sqm each)
                apt_wastage_pct = 4 + np.random.random() * 3  # Random wastage between 4-7%
                apt_wastage = apt_area * apt_wastage_pct / 100
                
                wastage_data.append({
                    'APARTMENT': apt_name,
                    'AREA (sqm)': round(apt_area, 2),
                    'WASTAGE (%)': round(apt_wastage_pct, 2),
                    'WASTAGE (sqm)': round(apt_wastage, 2)
                })
            
            # Add total row
            if wastage_data:
                total_area = sum(w['AREA (sqm)'] for w in wastage_data)
                total_wastage = sum(w['WASTAGE (sqm)'] for w in wastage_data)
                avg_wastage_pct = (total_wastage / total_area) * 100 if total_area > 0 else 0
                
                wastage_data.append({
                    'APARTMENT': 'TOTAL',
                    'AREA (sqm)': round(total_area, 2),
                    'WASTAGE (%)': round(avg_wastage_pct, 2),
                    'WASTAGE (sqm)': round(total_wastage, 2)
                })
            
            # Convert to DataFrame and save to Excel
            wastage_df = pd.DataFrame(wastage_data)
            wastage_df.to_excel(writer, sheet_name='1. Wastage Analysis', index=False)
            
            # Create summary sheet
            summary_data = []
            summary_data.append({'TYPE': 'Full Tiles', 'COUNT': full_tiles})
            summary_data.append({'TYPE': 'Irregular Tiles', 'COUNT': irregular_tiles})
            
            if has_pattern:
                summary_data.append({'TYPE': 'Cut X Tiles', 'COUNT': cut_x_tiles})
                summary_data.append({'TYPE': 'Cut Y Tiles', 'COUNT': cut_y_tiles})
            else:
                summary_data.append({'TYPE': 'All Cut Tiles', 'COUNT': all_cut_tiles})
            
            if small_tiles_count > 0:
                summary_data.append({'TYPE': 'Small Cut Tiles (excluded)', 'COUNT': small_tiles_count})
            
            summary_data.append({'TYPE': 'Total Tiles', 'COUNT': total_tiles})
            
            # Convert to DataFrame and save to Excel
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='2. Tile Summary', index=False)
            
            # Create area summary sheet by apartment
            area_data = []
            for apt_name in apartments:
                apt_tiles = [t for apt, tiles in apartments_data.items() 
                           for t in tiles['tiles'] if apt == apt_name]
                
                area_data.append({
                    'APARTMENT': apt_name,
                    'TILE COUNT': len(apt_tiles),
                    'AREA (sqm)': round(len(apt_tiles) * 0.36, 2)  # Assuming 600x600mm tiles
                })
            
            # Convert to DataFrame and save to Excel
            area_df = pd.DataFrame(area_data)
            area_df.to_excel(writer, sheet_name='3. Area Summary', index=False)
            
            # Create full tiles summary
            full_data = []
            for apt_name in apartments:
                apt_full_tiles = [t for apt, tiles in apartments_data.items() 
                               for t in tiles['tiles'] 
                               if apt == apt_name and t.get('classification') == 'full']
                
                full_data.append({
                    'APARTMENT': apt_name,
                    'FULL TILE COUNT': len(apt_full_tiles)
                })
            
            # Convert to DataFrame and save to Excel
            full_df = pd.DataFrame(full_data)
            full_df.to_excel(writer, sheet_name='4. Full Tiles', index=False)
            
            # Create cut tiles summaries based on pattern mode
            if has_pattern:
                # Cut X tiles
                cut_x_data = []
                for apt_name in apartments:
                    apt_cut_x_tiles = [t for apt, tiles in apartments_data.items() 
                                    for t in tiles['tiles'] 
                                    if apt == apt_name and t.get('classification') == 'cut_x']
                    
                    for tile in apt_cut_x_tiles:
                        cut_side = tile.get('cut_side', 0)
                        room_id = tile.get('room_id', -1)
                        
                        # Get room name
                        room_df = pd.DataFrame(session.get('room_df', []))
                        room_name = "Unknown"
                        if not room_df.empty:
                            room_row = room_df[room_df['room_id'] == room_id]
                            if not room_row.empty:
                                room_name = room_row.iloc[0]['room_name']
                        
                        cut_x_data.append({
                            'APARTMENT': apt_name,
                            'ROOM': room_name,
                            'CUT SIDE (mm)': round(cut_side),
                            'DIRECTION': 'X'
                        })
                
                # Group by apartment, room and cut side
                if cut_x_data:
                    cut_x_df = pd.DataFrame(cut_x_data)
                    cut_x_summary = cut_x_df.groupby(['APARTMENT', 'ROOM', 'CUT SIDE (mm)', 'DIRECTION']).size().reset_index(name='COUNT')
                    cut_x_summary.to_excel(writer, sheet_name='5. Cut X Tiles', index=False)
                
                # Cut Y tiles
                cut_y_data = []
                for apt_name in apartments:
                    apt_cut_y_tiles = [t for apt, tiles in apartments_data.items() 
                                    for t in tiles['tiles'] 
                                    if apt == apt_name and t.get('classification') == 'cut_y']
                    
                    for tile in apt_cut_y_tiles:
                        cut_side = tile.get('cut_side', 0)
                        room_id = tile.get('room_id', -1)
                        
                        # Get room name
                        room_df = pd.DataFrame(session.get('room_df', []))
                        room_name = "Unknown"
                        if not room_df.empty:
                            room_row = room_df[room_df['room_id'] == room_id]
                            if not room_row.empty:
                                room_name = room_row.iloc[0]['room_name']
                        
                        cut_y_data.append({
                            'APARTMENT': apt_name,
                            'ROOM': room_name,
                            'CUT SIDE (mm)': round(cut_side),
                            'DIRECTION': 'Y'
                        })
                
                # Group by apartment, room and cut side
                if cut_y_data:
                    cut_y_df = pd.DataFrame(cut_y_data)
                    cut_y_summary = cut_y_df.groupby(['APARTMENT', 'ROOM', 'CUT SIDE (mm)', 'DIRECTION']).size().reset_index(name='COUNT')
                    cut_y_summary.to_excel(writer, sheet_name='6. Cut Y Tiles', index=False)
            else:
                # All cut tiles
                all_cut_data = []
                for apt_name in apartments:
                    apt_all_cut_tiles = [t for apt, tiles in apartments_data.items() 
                                     for t in tiles['tiles'] 
                                     if apt == apt_name and t.get('classification') == 'all_cut']
                    
                    for tile in apt_all_cut_tiles:
                        cut_side = tile.get('cut_side', 0)
                        room_id = tile.get('room_id', -1)
                        
                        # Get room name
                        room_df = pd.DataFrame(session.get('room_df', []))
                        room_name = "Unknown"
                        if not room_df.empty:
                            room_row = room_df[room_df['room_id'] == room_id]
                            if not room_row.empty:
                                room_name = room_row.iloc[0]['room_name']
                        
                        all_cut_data.append({
                            'APARTMENT': apt_name,
                            'ROOM': room_name,
                            'CUT DIMENSION (mm)': round(cut_side)
                        })
                
                # Group by apartment, room and cut dimension
                if all_cut_data:
                    all_cut_df = pd.DataFrame(all_cut_data)
                    all_cut_summary = all_cut_df.groupby(['APARTMENT', 'ROOM', 'CUT DIMENSION (mm)']).size().reset_index(name='COUNT')
                    all_cut_summary.to_excel(writer, sheet_name='5. All Cut Tiles', index=False)
            
            # Create small cuts summary if available
            small_tiles_indices = small_cuts_data.get('small_tiles_indices', [])
            small_tiles_threshold = small_cuts_data.get('size_threshold', 10)
            
            if small_tiles_count > 0:
                small_cuts_data = []
                # Add summary row
                small_cuts_data.append({
                    'SIZE THRESHOLD': f"< {small_tiles_threshold}mm",
                    'COUNT': small_tiles_count,
                    'STATUS': 'Excluded from report'
                })
                
                small_cuts_df = pd.DataFrame(small_cuts_data)
                small_cuts_df.to_excel(writer, sheet_name='7. Small Cuts Summary', index=False)
        
        # Move to the beginning of the file
        output.seek(0)
        
        # Send the file for download
        return send_file(
            output,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error generating Excel file: {str(e)}"

# Helper function to generate a placeholder image
def generate_placeholder_image(title, width=800, height=600):
    """Generate a placeholder image with a title"""
    plt.figure(figsize=(width/100, height/100), dpi=100)
    plt.text(0.5, 0.5, title, ha='center', va='center', fontsize=24)
    plt.axis('off')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# Helper functions for serialization
def serialize_rooms(rooms):
    """Convert room polygons to a serializable format"""
    serialized_rooms = []
    for room in rooms:
        coords = list(room.exterior.coords)
        serialized_rooms.append({
            'coords': coords,
            'bounds': room.bounds
        })
    return serialized_rooms

# Replace the serialize_start_points function:
def serialize_start_points(start_points):
    """Convert start point data to a serializable format"""
    serialized_points = []
    for sp in start_points:
        serialized_points.append({
            'centroid': sp['centroid'],
            'width': sp['width'],
            'height': sp['height'],
            'area': sp.get('area', 0),
            'polygon_coords': list(sp['polygon'].exterior.coords)
        })
    return serialized_points

# Replace the deserialize_rooms function:
def deserialize_rooms(serialized_rooms):
    """Convert serialized room data back to Shapely polygons"""
    rooms = []
    for room_data in serialized_rooms:
        poly = Polygon(room_data['coords'])
        rooms.append(poly)
    return rooms

# Replace the deserialize_start_points function:
def deserialize_start_points(serialized_points):
    """Convert serialized start point data back to original format"""
    start_points = []
    for sp_data in serialized_points:
        start_points.append({
            'centroid': sp_data['centroid'],
            'width': sp_data['width'],
            'height': sp_data['height'],
            'area': sp_data.get('area', 0),
            'polygon': Polygon(sp_data['polygon_coords'])
        })
    return start_points



def generate_optimization_visualization(self, apartments_data, room_df, optimization_results):
    """
    Generate a visualization showing matched cut pieces
    
    Parameters:
    -----------
    apartments_data : dict
        Dictionary of apartment data
    room_df : DataFrame
        DataFrame containing room information
    optimization_results : dict
        Results from optimize_cut_pieces function
        
    Returns:
    --------
    str
        Base64 encoded image
    """
    plt.figure(figsize=(16, 12))
    
    # Define colors for different match types
    SAME_APT_COLOR = '#FFD700'  # Gold
    DIFF_APT_COLOR = '#FF6347'  # Tomato
    INV_COLOR = '#00BFFF'       # Deep Sky Blue
    
    # Plot room boundaries
    if isinstance(room_df, pd.DataFrame) and 'polygon' in room_df.columns:
        for _, room in room_df.iterrows():
            poly = room.get('polygon')
            if poly is not None and hasattr(poly, 'exterior'):
                x, y = poly.exterior.xy
                plt.plot(x, y, color='black', linewidth=1.5, alpha=0.8)
                
                # Add room label
                if 'apartment_name' in room and 'room_name' in room:
                    # Calculate centroid
                    center_x = sum(x) / len(x)
                    center_y = sum(y) / len(y)
                    
                    plt.text(center_x, center_y, f"{room['apartment_name']}-{room['room_name']}", 
                            fontsize=10, ha='center', va='center', 
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # Get tile polygons from apartments data
    tile_polygons = {}
    for apt_name, apt_data in apartments_data.items():
        tile_polygons[apt_name] = {}
        for i, tile in enumerate(apt_data.get('tiles', [])):
            if 'room_id' in tile:
                room_id = tile['room_id']
                if room_id not in tile_polygons[apt_name]:
                    tile_polygons[apt_name][room_id] = []
                
                # Store the tile index and polygon
                tile_info = {
                    'index': i,
                    'polygon': tile.get('polygon')
                }
                tile_polygons[apt_name][room_id].append(tile_info)
    
    # Process matches
    matches = optimization_results.get('matches', [])
    for match_idx, match in enumerate(matches):
        # Choose color based on match type
        if match.get('from_inventory', False):
            color = INV_COLOR
        elif match.get('cross_apartment', False):
            color = DIFF_APT_COLOR
        else:
            color = SAME_APT_COLOR
            
        # Process inventory matches differently
        if match.get('from_inventory', False):
            req = match.get('requirement', {})
            req_apt = req.get('apartment', '')
            req_room_id = req.get('room_id', -1)
            req_tile_idx = req.get('tile_idx', -1)
            
            # Find the tile in tile_polygons
            req_polygon = None
            if req_apt in tile_polygons and req_room_id in tile_polygons[req_apt]:
                for tile_info in tile_polygons[req_apt][req_room_id]:
                    if tile_info['index'] == req_tile_idx:
                        req_polygon = tile_info['polygon']
                        break
            
            # Draw the requirement tile if found
            if req_polygon and hasattr(req_polygon, 'exterior'):
                x, y = req_polygon.exterior.xy
                plt.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=1)
                
                # Add text showing it's an inventory match
                centroid = req_polygon.centroid
                plt.text(centroid.x, centroid.y, f"INV\n{req.get('dimension', 0):.0f}mm", 
                        fontsize=8, ha='center', va='center', color='white',
                        bbox=dict(facecolor=color, alpha=0.7, boxstyle='round,pad=0.1'))
        else:
            # Apartment-to-apartment match
            req = match.get('requirement', {})
            rem = match.get('remaining_piece', {})
            
            req_apt = req.get('apartment', '')
            req_room_id = req.get('room_id', -1)
            req_tile_idx = req.get('tile_idx', -1)
            
            rem_apt = rem.get('apartment', '')
            rem_room_id = rem.get('room_id', -1)
            rem_tile_idx = rem.get('tile_idx', -1)
            
            # Find the tiles in tile_polygons
            req_polygon = None
            rem_polygon = None
            
            if req_apt in tile_polygons and req_room_id in tile_polygons[req_apt]:
                for tile_info in tile_polygons[req_apt][req_room_id]:
                    if tile_info['index'] == req_tile_idx:
                        req_polygon = tile_info['polygon']
                        break
            
            if rem_apt in tile_polygons and rem_room_id in tile_polygons[rem_apt]:
                for tile_info in tile_polygons[rem_apt][rem_room_id]:
                    if tile_info['index'] == rem_tile_idx:
                        rem_polygon = tile_info['polygon']
                        break
            
            # Draw the tiles if found
            if req_polygon and hasattr(req_polygon, 'exterior'):
                x, y = req_polygon.exterior.xy
                plt.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=1)
            
            if rem_polygon and hasattr(rem_polygon, 'exterior'):
                x, y = rem_polygon.exterior.xy
                plt.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Draw a line connecting them if both found
            if req_polygon and rem_polygon and hasattr(req_polygon, 'centroid') and hasattr(rem_polygon, 'centroid'):
                req_centroid = req_polygon.centroid
                rem_centroid = rem_polygon.centroid
                
                plt.plot([req_centroid.x, rem_centroid.x], [req_centroid.y, rem_centroid.y], 
                        color=color, linewidth=2, alpha=0.8)
                
                # Add dimension label at midpoint
                mid_x = (req_centroid.x + rem_centroid.x) / 2
                mid_y = (req_centroid.y + rem_centroid.y) / 2
                plt.text(mid_x, mid_y, f"{req.get('dimension', 0):.0f}mm", 
                        fontsize=8, ha='center', va='center', color='white',
                        bbox=dict(facecolor=color, alpha=0.7, boxstyle='round,pad=0.1'))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SAME_APT_COLOR, alpha=0.7, edgecolor='black', label='Same Apartment Match'),
        Patch(facecolor=DIFF_APT_COLOR, alpha=0.7, edgecolor='black', label='Cross Apartment Match'),
        Patch(facecolor=INV_COLOR, alpha=0.7, edgecolor='black', label='Inventory Match')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add stats in a text box
    match_count = optimization_results.get('matched_count', 0)
    within_apt = optimization_results.get('within_apartment_matches', 0)
    cross_apt = optimization_results.get('cross_apartment_matches', 0)
    inv_matches = optimization_results.get('inventory_matches', 0)
    savings = optimization_results.get('total_savings', 0)
    
    stats_text = (
        f"Total Matches: {match_count}\n"
        f"Within-Apartment: {within_apt}\n"
        f"Cross-Apartment: {cross_apt}\n"
        f"Inventory: {inv_matches}\n"
        f"Material Saved: {savings:.0f} mm²"
    )
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    plt.title("Tile Cut Piece Optimization Matches")
    plt.axis('equal')
    plt.grid(False)
    plt.tight_layout()
    
    # Save figure to buffer and return base64 encoded string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/step8/analyze_cut_pieces', methods=['POST'])
def analyze_cut_pieces():
    """Step 8A: Analyze Cut Pieces from Export Data and Prepare for Matching"""
    try:
        # Get data from request
        data = request.get_json()
        
        # Get export data from step7
        export_info = session.get('export_info', {})
        
        # Get tile classification results
        tile_classification_results = session.get('classification_data', {})
        
        # Get small tiles results
        small_tiles_results = session.get('small_cuts_data', {})
        
        # Create optimization processor
        optimizer = OptimizationProcessor()
        
        # Convert export_info back to DataFrames
        processed_export_info = {}
        for key, value in export_info.items():
            if key in ['cut_x_simple', 'cut_y_simple', 'all_cut_simple'] and isinstance(value, list):
                processed_export_info[key] = pd.DataFrame(value)
            else:
                processed_export_info[key] = value
        
        # Analyze cut pieces
        cut_data_for_matching = optimizer.analyze_cut_pieces(
            processed_export_info, tile_classification_results, small_tiles_results)
        
        # Select apartments if provided
        selected_apartments = data.get('selected_apartments', [])
        if selected_apartments:
            # Filter cut data for selected apartments
            has_pattern = cut_data_for_matching.get('has_pattern', True)
            if has_pattern:
                # Filter X data
                cut_x_data = cut_data_for_matching.get('cut_x_data', pd.DataFrame())
                if not cut_x_data.empty and 'Apartment' in cut_x_data.columns:
                    cut_data_for_matching['cut_x_data'] = cut_x_data[cut_x_data['Apartment'].isin(selected_apartments)].copy()
                
                # Filter Y data
                cut_y_data = cut_data_for_matching.get('cut_y_data', pd.DataFrame())
                if not cut_y_data.empty and 'Apartment' in cut_y_data.columns:
                    cut_data_for_matching['cut_y_data'] = cut_y_data[cut_y_data['Apartment'].isin(selected_apartments)].copy()
            else:
                # Filter All data
                all_cut_data = cut_data_for_matching.get('all_cut_data', pd.DataFrame())
                if not all_cut_data.empty and 'Apartment' in all_cut_data.columns:
                    cut_data_for_matching['all_cut_data'] = all_cut_data[all_cut_data['Apartment'].isin(selected_apartments)].copy()
        
        # Create a serializable version of cut_data_for_matching
        serializable_data = {}
        for key, value in cut_data_for_matching.items():
            if isinstance(value, pd.DataFrame):
                serializable_data[key] = value.to_dict('records')
            else:
                serializable_data[key] = value
        
        # Store serializable version in session
        session['cut_data_for_matching'] = serializable_data
        
        return jsonify({
            'success': True,
            'cut_data': serializable_data
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error analyzing cut pieces: {str(e)}'})

@app.route('/step8/split_by_half', methods=['POST'])
def split_by_half():
    """Step 8B: Split Cut Pieces by Half-Tile Threshold"""
    try:
        # Get cut data from session
        cut_data_for_matching = session.get('cut_data_for_matching', {})
        
        if not cut_data_for_matching:
            return jsonify({'error': 'No cut data found. Please run Step 8A first.'})
        
        # Convert data from session back to DataFrames
        processed_data = {}
        for key, value in cut_data_for_matching.items():
            if key in ['cut_x_data', 'cut_y_data', 'all_cut_data'] and isinstance(value, list):
                processed_data[key] = pd.DataFrame(value)
            else:
                processed_data[key] = value
        
        # Create optimization processor
        optimizer = OptimizationProcessor()
        
        # Split cut pieces by half
        cut_pieces_by_half = optimizer.split_cut_pieces_by_half(processed_data)
        
        # Create a serializable version of cut_pieces_by_half
        serializable_pieces = {}
        for key, value in cut_pieces_by_half.items():
            if isinstance(value, pd.DataFrame):
                serializable_pieces[key] = value.to_dict('records')
            else:
                serializable_pieces[key] = value
        
        # Store serializable version in session
        session['cut_pieces_by_half'] = serializable_pieces
        
        return jsonify({
            'success': True,
            'cut_pieces_by_half': serializable_pieces
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error splitting cut pieces: {str(e)}'})

@app.route('/step8/match_cut_pieces', methods=['POST'])
def match_cut_pieces():
    """Step 8C: Match Cut Pieces with Inventory and Progressive Tolerance Ranges"""
    try:
        # Get data from request
        data = request.get_json()
        
        # Get cut pieces from session
        cut_pieces_by_half = session.get('cut_pieces_by_half', {})
        
        if not cut_pieces_by_half:
            return jsonify({'error': 'No cut pieces data found. Please run the previous steps first.'})
            
        # Create tolerance ranges from min to max
        min_tolerance = float(data.get('min_tolerance', 2.0))
        max_tolerance = float(data.get('max_tolerance', 80.0))
        
        # Use fixed tolerance ranges from Colab implementation
        tolerance_ranges = [10, 20, 40, 60, 80, 100]
        
        # Add more if max_tolerance is higher
        current_max = max(tolerance_ranges)
        while current_max < max_tolerance:
            current_max += 20
            tolerance_ranges.append(current_max)
            
        # Get inventory option
        use_inventory = data.get('use_inventory', False)
        
        # Get inventory data if enabled
        inventory_data = None
        if use_inventory and 'inventory_data' in session:
            print("Using inventory data for matching")
            inventory_data = session.get('inventory_data')
            
            # Debug the inventory data
            print(f"Inventory data keys: {list(inventory_data.keys() if inventory_data else [])}")
            for key, value in (inventory_data or {}).items():
                if isinstance(value, list) and len(value) > 0:
                    print(f"  {key}: {len(value)} items")
                    # Print first item
                    if len(value) > 0:
                        print(f"  First item: {value[0]}")
                elif key != 'has_pattern':
                    print(f"  {key}: {value}")
            
            # Check if pattern mode matches in both cut pieces and inventory
            if inventory_data and 'has_pattern' in inventory_data:
                inv_has_pattern = inventory_data.get('has_pattern', True)
                cut_has_pattern = cut_pieces_by_half.get('has_pattern', True)
                
                if inv_has_pattern != cut_has_pattern:
                    print(f"Warning: Pattern mode mismatch between inventory ({inv_has_pattern}) and cut pieces ({cut_has_pattern})")
        else:
            print("No inventory data used for matching")
        
        # Convert cut_pieces_by_half data to DataFrames
        processed_cut_pieces = {}
        has_pattern = cut_pieces_by_half.get('has_pattern', True)
        
        for key, value in cut_pieces_by_half.items():
            if isinstance(value, list) and key not in ['tolerance_ranges', 'half_threshold', 'tile_width', 'tile_height', 'has_pattern']:
                processed_cut_pieces[key] = pd.DataFrame(value)
            else:
                processed_cut_pieces[key] = value
        
        # Create optimization processor
        optimizer = OptimizationProcessor()
        
        # Call the processor to match cut pieces
        match_results = optimizer.match_cut_pieces(processed_cut_pieces, tolerance_ranges, inventory_data)
        
        # Store results in session
        session['match_results'] = make_serializable(match_results)
        
        # Get pattern mode for formatting the response
        has_pattern = cut_pieces_by_half.get('has_pattern', True)
        
        # Format results for UI display
        response_data = format_optimization_results(match_results, has_pattern)
        
        return jsonify({
            'success': True,
            'results': response_data
        })
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return jsonify({'error': f'Error matching cut pieces: {str(e)}'})


@app.route('/step8/export_matching_results', methods=['GET'])
def export_matching_results():
    """Export matching results to Excel file"""
    try:
        # Get match results from session
        match_results = session.get('match_results', {})
        
        if not match_results:
            return jsonify({'error': 'No matching results found. Please run optimization first.'})
        
        # Create optimization processor
        export_processor = ExportProcessor()
        
        # Export matching results
        output = export_processor.export_matching_results_to_excel(match_results)
        
        # Create response
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response.headers['Content-Disposition'] = 'attachment; filename=matching_results.xlsx'
        
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error exporting matching results: {str(e)}'})


@app.route('/step8/download_full_report', methods=['GET'])
def download_full_report():
    """Download a complete report package"""
    try:
        # Get match results from session
        match_results = session.get('match_results', {})
        
        if not match_results:
            return jsonify({'error': 'No matching results found. Please run optimization first.'})
        
        # Get inventory data if available
        inventory_data = session.get('inventory_data')
        
        # Create optimization processor
        export_processor = ExportProcessor()
        
        # Generate full report
        output = export_processor.download_full_report(match_results, inventory_data)
        
        # Create response
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'application/zip'
        response.headers['Content-Disposition'] = 'attachment; filename=optimization_report_package.zip'
        
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error generating full report: {str(e)}'})

def make_serializable(obj):
    """
    Recursively make a data structure JSON serializable by converting
    pandas DataFrames to lists of dictionaries and handling numpy types
    
    Parameters:
    -----------
    obj : any
        The object to make serializable
        
    Returns:
    --------
    A serializable version of the object
    """
    import numpy as np
    import pandas as pd
    from shapely.geometry import Polygon, MultiPolygon, Point, LineString
    
    # Handle None
    if obj is None:
        return None
    
    # Handle numpy types
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return make_serializable(obj.tolist())
    
    # Handle pandas DataFrame
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    
    # Handle pandas Series
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    
    # Handle Shapely geometry objects
    if isinstance(obj, (Polygon, MultiPolygon, Point, LineString)):
        return None
    
    # Handle lists
    if isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    
    # Handle other types (strings, etc.)
    return obj

def format_optimization_results(match_results, has_pattern=True):
    """
    Format match results for the web UI response with improved handling
    of different match types and unmatched pieces
    
    Parameters:
    -----------
    match_results : dict
        Dictionary containing the optimization results
    has_pattern : bool
        Whether the optimization used pattern-based tile classification
        
    Returns:
    --------
    dict
        Formatted results for UI display
    """
    import pandas as pd
    
    # Calculate statistics - ensure values are integers for JSON
    within_apartment_matches = int(match_results.get('within_apartment_matches', 0))
    cross_apartment_matches = int(match_results.get('cross_apartment_matches', 0))
    inventory_matches = int(match_results.get('inventory_matches', 0))
    total_savings = float(match_results.get('total_savings', 0))
    total_matches = within_apartment_matches + cross_apartment_matches + inventory_matches
    
    # Format simplified matches for display
    simplified_matches = match_results.get('simplified_matches', [])
    
    # Convert DataFrame to list if needed
    if isinstance(simplified_matches, pd.DataFrame):
        simplified_matches = simplified_matches.to_dict('records')
    elif isinstance(simplified_matches, dict) and 'From' in simplified_matches:
        # Handle single match record
        simplified_matches = [simplified_matches]
    
    # Create response data
    response_data = {
        'matched_count': total_matches,
        'within_apartment_matches': within_apartment_matches,
        'cross_apartment_matches': cross_apartment_matches,
        'inventory_matches': inventory_matches,
        'total_savings': float(total_savings),
        'optimization_plot': match_results.get('optimization_plot', ''),
        'has_pattern': has_pattern,
        'simplified_matches': simplified_matches
    }
    
    if has_pattern:
        # Split matches by direction
        x_matches = [m for m in simplified_matches if (isinstance(m, dict) and 
                                                     (m.get('Match ID', '').startswith(('X', 'IX', 'OX'))))]
        y_matches = [m for m in simplified_matches if (isinstance(m, dict) and 
                                                     (m.get('Match ID', '').startswith(('Y', 'IY', 'OY'))))]
        
        response_data['x_matches'] = x_matches
        response_data['y_matches'] = y_matches
        
        # Add unmatched pieces data - handle both DataFrame and list formats
        for key in ['x_unmatched_less', 'x_unmatched_more', 'y_unmatched_less', 'y_unmatched_more']:
            if key in match_results:
                if isinstance(match_results[key], pd.DataFrame) and not match_results[key].empty:
                    response_data[key] = match_results[key].to_dict('records')
                elif isinstance(match_results[key], list) and match_results[key]:
                    response_data[key] = match_results[key]
                else:
                    response_data[key] = []
    else:
        response_data['all_matches'] = simplified_matches
        
        # Add unmatched pieces data - handle both DataFrame and list formats
        for key in ['all_unmatched_less', 'all_unmatched_more']:
            if key in match_results:
                if isinstance(match_results[key], pd.DataFrame) and not match_results[key].empty:
                    response_data[key] = match_results[key].to_dict('records')
                elif isinstance(match_results[key], list) and match_results[key]:
                    response_data[key] = match_results[key]
                else:
                    response_data[key] = []
    
    return response_data


@app.route('/step8/full_process', methods=['POST'])
def step8_full_process():
    """Run the full optimization process in one go"""
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate data
        if not data:
            return jsonify({'error': 'Invalid data format'})
        
        # Get parameters
        selected_apartments = data.get('selected_apartments', [])
        min_tolerance = float(data.get('min_tolerance', 2.0))
        max_tolerance = float(data.get('max_tolerance', 80.0))
        prioritize_same_apartment = data.get('prioritize_same_apartment', True)
        use_inventory = data.get('use_inventory', False)
        
        # Ensure we have at least 2 apartments selected
        if len(selected_apartments) < 2:
            return jsonify({'error': 'Please select at least two apartments for optimization.'})
        
        # Check if we have necessary data in session
        if 'export_info' not in session:
            return jsonify({'error': 'No export data found. Please complete step 7 first.'})
        
        # Get data from session
        export_info = session.get('export_info', {})
        
        # Convert export_info back to DataFrames
        processed_export_info = {}
        for key, value in export_info.items():
            if key in ['cut_x_simple', 'cut_y_simple', 'all_cut_simple'] and isinstance(value, list):
                processed_export_info[key] = pd.DataFrame(value)
            else:
                processed_export_info[key] = value
        
        # Get classification data and small cuts data
        classification_data = session.get('classification_data', {})
        small_cuts_data = session.get('small_cuts_data', {})
        
        # Get inventory data if available and requested
        inventory_data = None
        if use_inventory and 'inventory_data' in session:
            inventory_data = session.get('inventory_data')
        
        # Create optimization processor
        optimizer = OptimizationProcessor()
        
        # Create tolerance ranges 
        tolerance_ranges = [10, 20, 40, 60, 80, 100]
        current_max = max(tolerance_ranges)
        while current_max < max_tolerance:
            current_max += 20  # Increment by 20mm
            tolerance_ranges.append(current_max)
        
        # STEP 8A: Analyze cut pieces
        cut_data_for_matching = optimizer.analyze_cut_pieces(
            processed_export_info, classification_data, small_cuts_data)
        
        # Filter cut data for selected apartments
        has_pattern = cut_data_for_matching.get('has_pattern', True)
        
        if has_pattern:
            # Filter X data
            cut_x_data = cut_data_for_matching.get('cut_x_data', pd.DataFrame())
            if not isinstance(cut_x_data, pd.DataFrame):
                cut_x_data = pd.DataFrame(cut_x_data) if cut_x_data else pd.DataFrame()
                
            if not cut_x_data.empty and 'Apartment' in cut_x_data.columns:
                cut_data_for_matching['cut_x_data'] = cut_x_data[cut_x_data['Apartment'].isin(selected_apartments)].copy()
            
            # Filter Y data
            cut_y_data = cut_data_for_matching.get('cut_y_data', pd.DataFrame())
            if not isinstance(cut_y_data, pd.DataFrame):
                cut_y_data = pd.DataFrame(cut_y_data) if cut_y_data else pd.DataFrame()
                
            if not cut_y_data.empty and 'Apartment' in cut_y_data.columns:
                cut_data_for_matching['cut_y_data'] = cut_y_data[cut_y_data['Apartment'].isin(selected_apartments)].copy()
        else:
            # Filter All data
            all_cut_data = cut_data_for_matching.get('all_cut_data', pd.DataFrame())
            if not isinstance(all_cut_data, pd.DataFrame):
                all_cut_data = pd.DataFrame(all_cut_data) if all_cut_data else pd.DataFrame()
                
            if not all_cut_data.empty and 'Apartment' in all_cut_data.columns:
                cut_data_for_matching['all_cut_data'] = all_cut_data[all_cut_data['Apartment'].isin(selected_apartments)].copy()
        
        # STEP 8B: Split cut pieces by half
        cut_pieces_by_half = optimizer.split_cut_pieces_by_half(cut_data_for_matching)
        
        # STEP 8C: Match cut pieces
        match_results = optimizer.match_cut_pieces(cut_pieces_by_half, tolerance_ranges, inventory_data)
        
        # Generate visualization 
        room_df = pd.DataFrame(session.get('room_df', []))
        room_polygons = deserialize_rooms(session.get('room_polygons', []))
        
        # Add polygons back to room_df for visualization
        if len(room_polygons) == len(room_df):
            room_df['polygon'] = room_polygons
            
            # Generate visualization
            try:
                match_visualization = generate_match_visualization(match_results, room_df, session)
                match_results['optimization_plot'] = match_visualization
            except Exception as viz_error:
                print(f"Error generating visualization: {viz_error}")
                import traceback
                traceback.print_exc()
        
        # Generate simplified matches for easier display and export
        export_processor = ExportProcessor()
        simplified_matches = export_processor.format_simplified_matches(match_results)
        match_results['simplified_matches'] = simplified_matches
        
        # Store results in session
        session['match_results'] = make_serializable(match_results)
        
        # Format results for response
        response_data = format_optimization_results(match_results, has_pattern)
        
        return jsonify({
            'success': True,
            'results': response_data
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error running full optimization process: {str(e)}'})

def generate_match_visualization(match_results, room_df, session=None):
    """
    Generate a basic tile layout visualization with match coloring.
    
    Parameters:
    -----------
    match_results : dict
        Dictionary with match results data
    room_df : DataFrame
        DataFrame with room information and polygons
    session : object, optional
        Flask session object to retrieve tile data
        
    Returns:
    --------
    str
        Base64 encoded image
    """
    import matplotlib.pyplot as plt
    import io
    import base64
    import matplotlib.patches as mpatches
    
    print("\n===== GENERATING MATCH VISUALIZATION =====")
    
    # Create figure with appropriate size
    plt.figure(figsize=(16, 12))
    
    # Define colors for different match types
    SAME_APT_COLORS = [
        '#FFD700', '#FFA500', '#FF6347', '#FF1493', '#9932CC', 
        '#4169E1', '#00BFFF', '#00FA9A', '#ADFF2F', '#FFD700'
    ]  # Multiple colors for same apartment matches
    DIFF_APT_COLOR = '#A9A9A9'  # Gray for different apartment matches
    INV_COLOR = '#8FBC8F'       # Green for inventory matches
    
    # Get simplified matches
    simplified_matches = match_results.get('simplified_matches', [])
    
    # Convert to list if it's a DataFrame
    if isinstance(simplified_matches, pd.DataFrame):
        simplified_matches = simplified_matches.to_dict('records')
    
    # Draw room outlines
    if 'polygon' in room_df.columns:
        for _, room in room_df.iterrows():
            poly = room.get('polygon')
            if poly is not None and hasattr(poly, 'exterior'):
                x, y = poly.exterior.xy
                plt.plot(x, y, color='black', linewidth=1.5, alpha=0.8)
                
                # Add room label if centroid coordinates are available
                if 'centroid_x' in room and 'centroid_y' in room:
                    plt.text(room['centroid_x'], room['centroid_y'], 
                            f"{room['apartment_name']}-{room['room_name']}", 
                            fontsize=10, ha='center', va='center', 
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # Create a map of match IDs to colors
    match_colors = {}
    same_apt_counter = 0
    
    for match in simplified_matches:
        match_id = match.get('Match ID', '')
        match_type = match.get('Match Type', '')
        
        # Determine color based on match type
        if match_type == 'Inventory' or match_id.startswith('I') or match_id.startswith('IX') or match_id.startswith('IY'):
            match_colors[match_id] = INV_COLOR
        elif match_type == 'Cross Apartment' or match_id.startswith('O') or match_id.startswith('OX') or match_id.startswith('OY'):
            match_colors[match_id] = DIFF_APT_COLOR
        else:  # Same apartment match
            match_num = int(re.search(r'\d+', match_id).group()) if re.search(r'\d+', match_id) else same_apt_counter
            color_idx = match_num % len(SAME_APT_COLORS)
            match_colors[match_id] = SAME_APT_COLORS[color_idx]
            same_apt_counter += 1
    
    # Draw matches as connected points
    match_pairs = []
    for match in simplified_matches:
        match_id = match.get('Match ID', '')
        from_str = match.get('From', '')
        to_str = match.get('To', '')
        
        # Skip if missing information
        if not match_id or not from_str or not to_str:
            continue
        
        # Get color
        color = match_colors.get(match_id, 'gray')
        
        # Create a unique pair identifier
        pair_id = f"{from_str}-{to_str}"
        match_pairs.append((pair_id, color, match_id))
    
    # Plot dummy points for the legend
    legend_elements = [
        mpatches.Patch(facecolor=SAME_APT_COLORS[0], alpha=0.7, edgecolor='black', label='Same Apartment Match'),
        mpatches.Patch(facecolor=DIFF_APT_COLOR, alpha=0.7, edgecolor='black', label='Cross Apartment Match'),
        mpatches.Patch(facecolor=INV_COLOR, alpha=0.7, edgecolor='black', label='Inventory Match')
    ]
    
    # Add a text box with statistics
    match_count = match_results.get('matched_count', 0)
    within_apt = match_results.get('within_apartment_matches', 0)
    cross_apt = match_results.get('cross_apartment_matches', 0)
    inv_matches = match_results.get('inventory_matches', 0)
    savings = match_results.get('total_savings', 0)
    
    stats_text = (
        f"Total Matches: {match_count}\n"
        f"Within-Apartment: {within_apt}\n"
        f"Cross-Apartment: {cross_apt}\n"
        f"Inventory: {inv_matches}\n"
        f"Material Saved: {savings:.0f} mm²"
    )
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title("Tile Cut Piece Optimization Matches")
    plt.axis('equal')
    plt.grid(False)
    plt.tight_layout()
    
    # Save figure to buffer and return base64 encoded string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/step8/debug_data', methods=['GET'])
def debug_step8_data():
    """Debug route to check what data is available for Step 8"""
    
    debug_info = {
        'export_info_available': 'export_info' in session,
        'classification_data_available': 'classification_data' in session,
        'small_cuts_data_available': 'small_cuts_data' in session,
        'inventory_data_available': 'inventory_data' in session,
        'cut_data_for_matching_available': 'cut_data_for_matching' in session,
        'cut_pieces_by_half_available': 'cut_pieces_by_half' in session,
        'match_results_available': 'match_results' in session
    }
    
    # Get key stats if available
    if 'export_info' in session:
        export_info = session.get('export_info', {})
        debug_info['export_info_keys'] = list(export_info.keys())
        
        # Check if we have the required cut data
        if 'cut_x_simple' in export_info:
            if isinstance(export_info['cut_x_simple'], list):
                debug_info['cut_x_simple_length'] = len(export_info['cut_x_simple'])
            elif isinstance(export_info['cut_x_simple'], pd.DataFrame):
                debug_info['cut_x_simple_length'] = len(export_info['cut_x_simple'])
        
        if 'cut_y_simple' in export_info:
            if isinstance(export_info['cut_y_simple'], list):
                debug_info['cut_y_simple_length'] = len(export_info['cut_y_simple'])
            elif isinstance(export_info['cut_y_simple'], pd.DataFrame):
                debug_info['cut_y_simple_length'] = len(export_info['cut_y_simple'])
            
        if 'all_cut_simple' in export_info:
            if isinstance(export_info['all_cut_simple'], list):
                debug_info['all_cut_simple_length'] = len(export_info['all_cut_simple'])
            elif isinstance(export_info['all_cut_simple'], pd.DataFrame):
                debug_info['all_cut_simple_length'] = len(export_info['all_cut_simple'])
    
    if 'classification_data' in session:
        classification_data = session.get('classification_data', {})
        debug_info['has_pattern'] = classification_data.get('has_pattern', None)
        debug_info['classification_keys'] = list(classification_data.keys())
    
    if 'room_df' in session:
        room_df = session.get('room_df', [])
        debug_info['room_df_length'] = len(room_df)
        if len(room_df) > 0:
            debug_info['apartments'] = list(set([r.get('apartment_name', '') for r in room_df]))
    
    return jsonify(debug_info)

def generate_apartment_preview(match_results, apartment):
    """
    Generate a preview of apartment-specific matches for the UI
    
    Parameters:
    -----------
    match_results : dict
        Dictionary with match results data
    apartment : str
        Apartment ID to generate preview for
        
    Returns:
    --------
    dict
        Preview data for UI display
    """
    import pandas as pd
    
    # Extract matches for this apartment
    simplified_matches = match_results.get('simplified_matches', [])
    
    # Convert to list if it's a DataFrame
    if isinstance(simplified_matches, pd.DataFrame):
        simplified_matches = simplified_matches.to_dict('records')
    
    # Filter matches for this apartment
    apt_matches = []
    for match in simplified_matches:
        from_apt = match.get('From', '').split('-')[0] if '-' in match.get('From', '') else ''
        to_apt = match.get('To', '').split('-')[0] if '-' in match.get('To', '') else ''
        
        if from_apt == apartment or to_apt == apartment:
            # Make sure the match has all required fields
            formatted_match = {
                'Match_ID': match.get('Match ID', ''),
                'From': match.get('From', ''),
                'To': match.get('To', ''),
                'Size_mm': match.get('Size (mm)', 0),
                'Match_Type': match.get('Match Type', '')
            }
            apt_matches.append(formatted_match)
    
    # Categorize matches by type
    match_types = {
        'same_apt': [],
        'cross_apt': [],
        'inventory': []
    }
    
    # Process each match and categorize by type
    for match in apt_matches:
        match_type = match['Match_Type']
        match_id = match['Match_ID']
        
        # Inventory matches
        if match_type == 'Inventory' or match_id.startswith(('IX', 'IY', 'I')):
            match_types['inventory'].append(match)
        
        # Same apartment matches
        elif match_type == 'Same Apartment' or (match_id.startswith(('X', 'Y')) and not match_id.startswith(('OX', 'OY'))):
            match_types['same_apt'].append(match)
        
        # Cross apartment matches
        elif match_type == 'Cross Apartment' or match_id.startswith(('OX', 'OY')):
            match_types['cross_apt'].append(match)
        
        # Unknown type (fallback)
        else:
            # Try to determine type from the From/To fields
            from_apt = match['From'].split('-')[0] if '-' in match['From'] else ''
            to_apt = match['To'].split('-')[0] if '-' in match['To'] else ''
            
            if from_apt == to_apt:
                match_types['same_apt'].append(match)
            else:
                match_types['cross_apt'].append(match)
    
    # Get statistics
    has_pattern = match_results.get('has_pattern', True)
    
    # Create preview data
    preview_data = {
        'apartment': apartment,
        'matches': apt_matches,
        'has_pattern': has_pattern,
        'same_apt_count': len(match_types['same_apt']),
        'cross_apt_count': len(match_types['cross_apt']),
        'inv_count': len(match_types['inventory']),
        'total_matches': len(apt_matches)
    }
    
    # Add direction-specific data if using pattern
    if has_pattern:
        x_matches = [m for m in apt_matches if m['Match_ID'].startswith(('X', 'IX', 'OX'))]
        y_matches = [m for m in apt_matches if m['Match_ID'].startswith(('Y', 'IY', 'OY'))]
        
        preview_data['x_matches'] = x_matches
        preview_data['y_matches'] = y_matches
        preview_data['x_matches_count'] = len(x_matches)
        preview_data['y_matches_count'] = len(y_matches)
    
    # Calculate material savings for this apartment
    # We need to calculate this carefully based on match types
    total_savings = 0
    
    # Full credit for within-apartment and inventory matches
    for match in match_types['same_apt'] + match_types['inventory']:
        total_savings += float(match.get('Size_mm', 0))
    
    # Half credit for cross-apartment matches (to avoid double counting)
    for match in match_types['cross_apt']:
        total_savings += float(match.get('Size_mm', 0)) / 2
    
    preview_data['material_saved_mm2'] = total_savings
    preview_data['material_saved_m2'] = total_savings / 1000000
    
    return preview_data

def extract_apartment_from_match_id(match_id_string):
    """
    Extract apartment name from a match ID string in format 'Apt-Room-...'
    
    Parameters:
    -----------
    match_id_string : str
        The match ID string to parse
        
    Returns:
    --------
    str or None
        The extracted apartment name, or None if not found
    """
    if not match_id_string or not isinstance(match_id_string, str):
        return None
        
    # Check if string contains a dash
    if '-' in match_id_string:
        # Extract the part before the first dash
        return match_id_string.split('-')[0]
    
    return None

@app.route('/step9', methods=['GET', 'POST'])
def step9():
    """Step 9: Detailed Export Reports"""
    if request.method == 'GET':
        try:
            # Check if match_results exists in session
            if 'match_results' not in session:
                # Use flash instead of failing silently
                flash('Please complete the optimization step first.', 'warning')
                return redirect(url_for('step8'))
            
            # Get optimization statistics for display
            match_results = session.get('match_results', {})
            
            # Basic stats for the template
            match_stats = {
                'total_matches': match_results.get('matched_count', 0),
                'within_apartment_matches': match_results.get('within_apartment_matches', 0),
                'cross_apartment_matches': match_results.get('cross_apartment_matches', 0),
                'inventory_matches': match_results.get('inventory_matches', 0),
                'total_savings': float(match_results.get('total_savings', 0))
            }
            
            # Get apartment list for display
            apartments = []
            room_df = pd.DataFrame(session.get('room_df', []))
            if not room_df.empty and 'apartment_name' in room_df.columns:
                apartments = sorted(room_df['apartment_name'].unique())
            
            # Get apartment-specific match statistics using simplified_matches
            apartment_stats = {}
            
            # Initialize stats for all apartments
            for apt in apartments:
                apartment_stats[apt] = {
                    'match_count': 0,
                    'within_apartment': 0,
                    'cross_apartment': 0,
                    'inventory': 0
                }
            
            # Extract simplified matches
            simplified_matches = match_results.get('simplified_matches', [])
            
            # Convert to list if it's a DataFrame
            if isinstance(simplified_matches, pd.DataFrame):
                simplified_matches = simplified_matches.to_dict('records')
            
            # Count matches per apartment (COUNT EACH MATCH ONLY ONCE)
            total_counted = 0
            for match in simplified_matches:
                match_type = match.get('Match Type', '')
                
                # Determine which apartment(s) to count this match for
                if match_type == 'Inventory':
                    # For inventory matches, count only for the apartment using inventory
                    if 'From' in match and '-' in match['From']:
                        apt = match['From'].split('-')[0]
                        if apt in apartment_stats:
                            apartment_stats[apt]['match_count'] += 1
                            apartment_stats[apt]['inventory'] += 1
                            total_counted += 1
                
                elif match_type == 'Same Apartment':
                    # For same-apartment matches, count for that one apartment
                    if 'From' in match and '-' in match['From']:
                        apt = match['From'].split('-')[0]
                        if apt in apartment_stats:
                            apartment_stats[apt]['match_count'] += 1
                            apartment_stats[apt]['within_apartment'] += 1
                            total_counted += 1
                
                elif match_type == 'Cross Apartment':
                    # For cross-apartment matches, count half for each apartment
                    from_apt = None
                    to_apt = None
                    
                    if 'From' in match and '-' in match['From']:
                        from_apt = match['From'].split('-')[0]
                    
                    if 'To' in match and '-' in match['To']:
                        to_apt = match['To'].split('-')[0]
                    
                    # Add 0.5 matches to each apartment (will round up when displaying)
                    if from_apt and from_apt in apartment_stats:
                        apartment_stats[from_apt]['match_count'] += 0.5
                        apartment_stats[from_apt]['cross_apartment'] += 0.5
                    
                    if to_apt and to_apt in apartment_stats:
                        apartment_stats[to_apt]['match_count'] += 0.5
                        apartment_stats[to_apt]['cross_apartment'] += 0.5
                    
                    total_counted += 1
                
                # Fallback for matches without explicit type but with Match ID
                elif 'Match ID' in match:
                    match_id = match['Match ID']
                    
                    # Inventory match (IX, IY)
                    if match_id.startswith(('IX', 'IY', 'I')):
                        if 'From' in match and '-' in match['From']:
                            apt = match['From'].split('-')[0]
                            if apt in apartment_stats:
                                apartment_stats[apt]['match_count'] += 1
                                apartment_stats[apt]['inventory'] += 1
                                total_counted += 1
                    
                    # Same apartment match (X, Y)
                    elif match_id.startswith(('X', 'Y')) and not match_id.startswith(('OX', 'OY')):
                        if 'From' in match and '-' in match['From']:
                            apt = match['From'].split('-')[0]
                            if apt in apartment_stats:
                                apartment_stats[apt]['match_count'] += 1
                                apartment_stats[apt]['within_apartment'] += 1
                                total_counted += 1
                    
                    # Cross apartment match (OX, OY)
                    elif match_id.startswith(('OX', 'OY')):
                        from_apt = None
                        to_apt = None
                        
                        if 'From' in match and '-' in match['From']:
                            from_apt = match['From'].split('-')[0]
                        
                        if 'To' in match and '-' in match['To']:
                            to_apt = match['To'].split('-')[0]
                        
                        # Add 0.5 matches to each apartment
                        if from_apt and from_apt in apartment_stats:
                            apartment_stats[from_apt]['match_count'] += 0.5
                            apartment_stats[from_apt]['cross_apartment'] += 0.5
                        
                        if to_apt and to_apt in apartment_stats:
                            apartment_stats[to_apt]['match_count'] += 0.5
                            apartment_stats[to_apt]['cross_apartment'] += 0.5
                        
                        total_counted += 1
            
            # Round the counts for display (we used 0.5 for cross-apartment matches)
            for apt in apartment_stats:
                apartment_stats[apt]['match_count'] = int(round(apartment_stats[apt]['match_count']))
                apartment_stats[apt]['within_apartment'] = int(round(apartment_stats[apt]['within_apartment']))
                apartment_stats[apt]['cross_apartment'] = int(round(apartment_stats[apt]['cross_apartment']))
                apartment_stats[apt]['inventory'] = int(apartment_stats[apt]['inventory'])
            
            # For debugging - ensure total count matches
            print(f"Total matches in match_results: {match_stats['total_matches']}")
            print(f"Total matches counted for apartments: {total_counted}")
            sum_apt_matches = sum(apt_stats['match_count'] for apt_stats in apartment_stats.values())
            print(f"Sum of all apartment match counts: {sum_apt_matches}")
            
            # Get optimization plot if available
            optimization_plot = match_results.get('optimization_plot', None)
            
            return render_template('step9.html',
                                  match_stats=match_stats,
                                  apartments=apartments,
                                  apartment_stats=apartment_stats,
                                  optimization_plot=optimization_plot)
        except Exception as e:
            print(f"Error in step9 GET: {str(e)}")
            import traceback
            traceback.print_exc()
            # Use flash instead of failing silently
            flash('An error occurred. Please try again.', 'error')
            return redirect(url_for('step8'))
    else:
        # POST method is handled by the /step9/generate_reports endpoint
        return jsonify({'error': 'Direct POST to /step9 is not supported. Please use /step9/generate_reports instead.'})

@app.route('/step9/generate_reports', methods=['POST'])
def generate_reports():
    """Generate detailed reports based on matching results"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Invalid request data'})
            
        # Get required parameters
        project_name = data.get('project_name', 'Tile_Project')
        include_apartment_reports = data.get('include_apartment_reports', True)
        include_inventory_report = data.get('include_inventory_report', True)
        include_summary_report = data.get('include_summary_report', True)
        export_format = data.get('export_format', 'zip')
        include_visualization = data.get('include_visualization', True)
        
        # Check if match_results exists in session
        if 'match_results' not in session:
            return jsonify({'error': 'No match results found. Please run the optimization process first.'})
        
        # Get match results from session
        match_results = session.get('match_results', {})
        
        # Get pattern mode and has_inventory
        has_pattern = match_results.get('has_pattern', True)
        has_inventory = match_results.get('inventory_matches', 0) > 0
        
        # Create export processor instance
        export_processor = ExportProcessor()
        
        # Initialize export folder with appropriate name
        export_folder = f"{project_name}_Reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        export_processor.export_folder = os.path.join('exports', export_folder)
        os.makedirs(export_processor.export_folder, exist_ok=True)
        
        # Run full export process
        export_result = export_processor.export_full_report(match_results, has_pattern, has_inventory)
        
        # Prepare file information for response
        files = []
        
        # Add summary file if available
        if export_result['summary_file'] and include_summary_report:
            files.append({
                'name': os.path.basename(export_result['summary_file']),
                'type': 'excel',
                'size': f"{os.path.getsize(export_result['summary_file'])/1024:.1f} KB"
            })
        
        # Add apartment files if available
        if include_apartment_reports:
            for apt_file in export_result['apt_files']:
                files.append({
                    'name': os.path.basename(apt_file),
                    'type': 'excel',
                    'size': f"{os.path.getsize(apt_file)/1024:.1f} KB"
                })
        
        # Add inventory file if available
        if export_result['inv_file'] and include_inventory_report:
            files.append({
                'name': os.path.basename(export_result['inv_file']),
                'type': 'excel',
                'size': f"{os.path.getsize(export_result['inv_file'])/1024:.1f} KB"
            })
        
        # Set download URL based on export format
        if export_format == 'zip' and export_result['zip_file']:
            download_url = f"/downloads/{os.path.basename(export_result['zip_file'])}"
        else:
            download_url = f"/step9/download_all_reports"
        
        return jsonify({
            'success': True,
            'download_url': download_url,
            'files': files
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error generating reports: {str(e)}'})

@app.route('/step9/download_report', methods=['GET'])
def download_report():
    """Download a specific report"""
    try:
        # Get parameters
        report_type = request.args.get('type', 'summary')
        apartment = request.args.get('apartment', None)
        
        # Check if match_results exists in session
        if 'match_results' not in session:
            return "No match results found. Please run the optimization process first."
        
        # Get match results
        match_results = session.get('match_results', {})
        
        # Create export processor
        export_processor = ExportProcessor()
        
        # Generate the requested report
        if report_type == 'summary':
            # Create a summary report
            has_pattern = match_results.get('has_pattern', True)
            has_inventory = match_results.get('inventory_matches', 0) > 0
            
            # Initialize export folder
            export_folder = export_processor.initialize_export_folder()
            
            # Create summary data
            count_summary = {
                'total_matches': match_results.get('matched_count', 0),
                'within_apartment_matches': match_results.get('within_apartment_matches', 0),
                'cross_apartment_matches': match_results.get('cross_apartment_matches', 0),
                'inventory_matches': match_results.get('inventory_matches', 0),
                'total_savings': match_results.get('total_savings', 0)
            }
            
            # Create temporary files to pass to the summary function
            apt_files = []
            inv_file = None
            
            # Create summary report
            summary_file = export_processor.create_summary_report(apt_files, inv_file, count_summary, has_pattern, has_inventory)
            
            # Read the file
            with open(summary_file, 'rb') as f:
                output = io.BytesIO(f.read())
            
            # Create response
            response = make_response(output.getvalue())
            response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            response.headers['Content-Disposition'] = f'attachment; filename=Summary_Report.xlsx'
            
            return response
        
        elif report_type == 'apartment' and apartment:
            # Extract apartment data from match results
            has_pattern = match_results.get('has_pattern', True)
            
            # Initialize export folder
            export_folder = export_processor.initialize_export_folder()
            
            # Get apartment raw data from the match results
            if has_pattern:
                # Combine X and Y data for this apartment
                x_apt_df = match_results.get('x_report', {}).get('raw_df', pd.DataFrame())
                y_apt_df = match_results.get('y_report', {}).get('raw_df', pd.DataFrame())
                apt_raw_df = pd.concat([x_apt_df, y_apt_df]) if not x_apt_df.empty or not y_apt_df.empty else pd.DataFrame()
            else:
                # All direction data
                apt_raw_df = match_results.get('all_report', {}).get('raw_df', pd.DataFrame())
            
            # Filter and export apartment data
            apt_file = export_processor.export_apartment_data(apt_raw_df, apartment, has_pattern)
            
            if apt_file:
                # Read the file
                with open(apt_file, 'rb') as f:
                    output = io.BytesIO(f.read())
                
                # Create response
                response = make_response(output.getvalue())
                response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                response.headers['Content-Disposition'] = f'attachment; filename={apartment}_Report.xlsx'
                
                return response
            else:
                return f"No data found for apartment {apartment}"
        
        elif report_type == 'inventory':
            # Get inventory data from match results
            has_pattern = match_results.get('has_pattern', True)
            
            # Initialize export folder
            export_folder = export_processor.initialize_export_folder()
            
            # Get inventory data
            if has_pattern:
                x_inv_df = match_results.get('x_inv_report', pd.DataFrame())
                y_inv_df = match_results.get('y_inv_report', pd.DataFrame())
                inv_df = pd.concat([x_inv_df, y_inv_df]) if not x_inv_df.empty or not y_inv_df.empty else pd.DataFrame()
            else:
                inv_df = match_results.get('all_inv_report', pd.DataFrame())
            
            # Get apartment data for unmatched pieces
            if has_pattern:
                x_apt_df = match_results.get('x_report', {}).get('raw_df', pd.DataFrame())
                y_apt_df = match_results.get('y_report', {}).get('raw_df', pd.DataFrame())
                apt_raw_df = pd.concat([x_apt_df, y_apt_df]) if not x_apt_df.empty or not y_apt_df.empty else pd.DataFrame()
            else:
                apt_raw_df = match_results.get('all_report', {}).get('raw_df', pd.DataFrame())
            
            # Export inventory data
            inv_file = export_processor.export_inventory_data(inv_df, apt_raw_df, has_pattern)
            
            if inv_file:
                # Read the file
                with open(inv_file, 'rb') as f:
                    output = io.BytesIO(f.read())
                
                # Create response
                response = make_response(output.getvalue())
                response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                response.headers['Content-Disposition'] = f'attachment; filename=Inventory_Report.xlsx'
                
                return response
            else:
                return "No inventory data found"
        
        elif report_type == 'visualization':
            # Get visualization image from match results
            if 'optimization_plot' not in match_results or not match_results['optimization_plot']:
                return "No visualization available."
            
            # Decode base64 image
            import base64
            
            # Try to convert the base64 string to binary
            try:
                if match_results['optimization_plot'].startswith('data:image'):
                    # Handle data URI
                    header, encoded = match_results['optimization_plot'].split(",", 1)
                    img_data = base64.b64decode(encoded)
                else:
                    # Handle raw base64
                    img_data = base64.b64decode(match_results['optimization_plot'])
                
                # Create response
                response = make_response(img_data)
                response.headers['Content-Type'] = 'image/png'
                response.headers['Content-Disposition'] = f'attachment; filename=optimization_visualization.png'
                
                return response
            except:
                return "Error decoding visualization data"
        else:
            return "Invalid report type specified."
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error generating report: {str(e)}"

@app.route('/step9/download_all_reports', methods=['GET'])
def download_all_reports():
    """Download all reports as a single ZIP package"""
    try:
        # Check if match_results exists in session
        if 'match_results' not in session:
            return "No match results found. Please run the optimization process first."
        
        # Get match results
        match_results = session.get('match_results', {})
        
        # Get inventory data if available
        inventory_data = session.get('inventory_data', None)
        
        # Create export processor
        export_processor = ExportProcessor()
        
        # Generate full report package
        output = export_processor.download_full_report(match_results, inventory_data)
        
        # Create response
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'application/zip'
        response.headers['Content-Disposition'] = 'attachment; filename=optimization_reports.zip'
        
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error generating report package: {str(e)}"

@app.route('/step9/preview_report', methods=['GET'])
def preview_report():
    """Generate a preview of a specific report for display in the UI"""
    try:
        # Get parameters
        report_type = request.args.get('type', 'summary')
        apartment = request.args.get('apartment', None)
        
        # Check if match_results exists in session
        if 'match_results' not in session:
            return jsonify({'error': 'No match results found. Please run the optimization process first.'})
        
        # Get match results
        match_results = session.get('match_results', {})
        
        # Generate preview data based on report type
        if report_type == 'summary':
            # For summary preview, return basic statistics
            preview_data = {
                'total_matches': match_results.get('matched_count', 0),
                'within_apartment_matches': match_results.get('within_apartment_matches', 0),
                'cross_apartment_matches': match_results.get('cross_apartment_matches', 0),
                'inventory_matches': match_results.get('inventory_matches', 0),
                'total_savings': float(match_results.get('total_savings', 0)),
                'has_pattern': match_results.get('has_pattern', True)
            }
        elif report_type == 'apartment' and apartment:
            # Extract apartment-specific matches
            simplified_matches = match_results.get('simplified_matches', [])
            
            # Convert to list if it's a DataFrame
            if isinstance(simplified_matches, pd.DataFrame):
                simplified_matches = simplified_matches.to_dict('records')
            
            # Filter matches for this apartment
            apt_matches = []
            for match in simplified_matches:
                from_apt = match.get('From', '').split('-')[0] if '-' in match.get('From', '') else ''
                to_apt = match.get('To', '').split('-')[0] if '-' in match.get('To', '') else ''
                
                if from_apt == apartment or to_apt == apartment:
                    # Make sure the match has all required fields
                    formatted_match = {
                        'Match_ID': match.get('Match ID', ''),
                        'From': match.get('From', ''),
                        'To': match.get('To', ''),
                        'Size_mm': match.get('Size (mm)', 0),
                        'Match_Type': match.get('Match Type', '')
                    }
                    apt_matches.append(formatted_match)
            
            # Create preview data
            preview_data = {
                'apartment': apartment,
                'matches': apt_matches,
                'has_pattern': match_results.get('has_pattern', True),
                'same_apt_count': len([m for m in apt_matches if m['Match_Type'] == 'Same Apartment']),
                'cross_apt_count': len([m for m in apt_matches if m['Match_Type'] == 'Cross Apartment']),
                'inv_count': len([m for m in apt_matches if m['Match_Type'] == 'Inventory']),
                'total_matches': len(apt_matches)
            }
            
            # Add direction-specific data if using pattern
            if preview_data['has_pattern']:
                x_matches = [m for m in apt_matches if m['Match_ID'].startswith(('X', 'IX', 'OX'))]
                y_matches = [m for m in apt_matches if m['Match_ID'].startswith(('Y', 'IY', 'OY'))]
                
                preview_data['x_matches'] = x_matches
                preview_data['y_matches'] = y_matches
                preview_data['x_matches_count'] = len(x_matches)
                preview_data['y_matches_count'] = len(y_matches)
        
        elif report_type == 'inventory':
            # For inventory preview
            inventory_data = session.get('inventory_data', None)
            
            preview_data = {
                'has_inventory': 'inventory_matches' in match_results and match_results['inventory_matches'] > 0,
                'inventory_matches': match_results.get('inventory_matches', 0)
            }
            
            # Get inventory-specific matches
            if preview_data['has_inventory']:
                inventory_matches = []
                simplified_matches = match_results.get('simplified_matches', [])
                
                # Convert to list if it's a DataFrame
                if isinstance(simplified_matches, pd.DataFrame):
                    simplified_matches = simplified_matches.to_dict('records')
                
                for match in simplified_matches:
                    match_id = match.get('Match ID', '')
                    match_type = match.get('Match Type', '')
                    
                    if match_type == 'Inventory' or match_id.startswith('I') or match_id.startswith('IX') or match_id.startswith('IY'):
                        inventory_matches.append(match)
                
                preview_data['inventory_matches_data'] = inventory_matches
                
                # Add inventory stats if available
                if inventory_data:
                    has_pattern_inv = inventory_data.get('has_pattern', True)
                    
                    # Count inventory items
                    total_items = 0
                    if has_pattern_inv:
                        x_inv = inventory_data.get('cut_x_inventory', [])
                        y_inv = inventory_data.get('cut_y_inventory', [])
                        
                        if isinstance(x_inv, list):
                            total_items += len(x_inv)
                        elif isinstance(x_inv, pd.DataFrame):
                            total_items += len(x_inv)
                            
                        if isinstance(y_inv, list):
                            total_items += len(y_inv)
                        elif isinstance(y_inv, pd.DataFrame):
                            total_items += len(y_inv)
                    else:
                        all_inv = inventory_data.get('all_cut_inventory', [])
                        
                        if isinstance(all_inv, list):
                            total_items += len(all_inv)
                        elif isinstance(all_inv, pd.DataFrame):
                            total_items += len(all_inv)
                    
                    preview_data['total_inventory_items'] = total_items
                    preview_data['pattern_mode'] = has_pattern_inv
        else:
            return jsonify({'error': 'Invalid report type specified.'})
        
        return jsonify({
            'success': True,
            'preview': preview_data
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error generating preview: {str(e)}'})


if __name__ == '__main__':
    app.run(debug=True)
