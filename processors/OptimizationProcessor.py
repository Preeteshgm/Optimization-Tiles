import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from shapely.geometry import Polygon, Point, MultiPolygon

class OptimizationProcessor:
    """Processor for optimizing cut piece usage across apartments and with inventory"""
    
    def __init__(self):
        """Initialize the optimization processor"""
        pass
    
    def analyze_cut_pieces(self, export_info, tile_classification_results, small_tiles_results):
        """
        Analyze cut pieces from Export Data and prepare for matching
        
        Parameters:
        - export_info: Dict with export data from Step 7
        - tile_classification_results: Dict with tile classification data
        - small_tiles_results: Dict with small tiles data
        
        Returns:
        - Dict with cut data formatted for matching
        """
        print("\n🧩 Step 8A: Analyze Cut Pieces from Export Data and Prepare for Matching")
        
        # Get pattern mode
        has_pattern = tile_classification_results.get('has_pattern', True)
        print(f"Pattern mode: {'With Pattern (separate X/Y cuts)' if has_pattern else 'No Pattern (all cuts)'}")
        
        # Get size threshold from small_tiles_results
        size_threshold = small_tiles_results.get('size_threshold', 10) if small_tiles_results else 10
        print(f"Small tiles threshold: {size_threshold}mm (tiles smaller than this were excluded)")
        
        # Get standard tile dimensions
        tiles_df = None
        if 'tiles_df' in tile_classification_results:
            if isinstance(tile_classification_results['tiles_df'], pd.DataFrame):
                tiles_df = tile_classification_results['tiles_df']
            else:
                # Try to convert to DataFrame if it's a list
                if isinstance(tile_classification_results['tiles_df'], list):
                    tiles_df = pd.DataFrame(tile_classification_results['tiles_df'])
        
        tile_width = 600
        tile_height = 600
        
        if tiles_df is not None and not tiles_df.empty:
            sample_tiles = []
            for _, tile in tiles_df.iterrows():
                if 'actual_width' in tile and 'actual_height' in tile:
                    if isinstance(tile['actual_width'], (int, float)) and isinstance(tile['actual_height'], (int, float)):
                        if tile['actual_width'] > 0 and tile['actual_height'] > 0:
                            sample_tiles.append(tile)
                            break
            
            if sample_tiles:
                standard_tile = sample_tiles[0]
                tile_width = standard_tile['actual_width']
                tile_height = standard_tile['actual_height']
                print(f"Standard tile size: {tile_width}mm x {tile_height}mm")
            else:
                print("⚠️ Could not determine standard tile size from sample tiles")
        else:
            print("⚠️ No tiles_df found in classification results, using default tile size")
        
        # Use the simplified cut data directly from export_info
        if has_pattern:
            # --- Process Cut X data ---
            print("\n📋 Cut X Data (with small tiles < {size_threshold}mm excluded):")
            
            # Handle different data formats - export_info might contain DataFrames or lists of dicts
            cut_x_simple = export_info.get('cut_x_simple', [])
            if isinstance(cut_x_simple, list):
                cut_x_simple = pd.DataFrame(cut_x_simple)
            
            if not isinstance(cut_x_simple, pd.DataFrame):
                cut_x_simple = pd.DataFrame()
            
            if not cut_x_simple.empty:
                # Verify column names exist
                required_cols = ['APPARTMENT NUMBER', 'CUT SIDE (mm)', 'LOCATION', 'COUNT']
                if not all(col in cut_x_simple.columns for col in required_cols):
                    # Try to find similar column names if exact ones don't exist
                    col_mapping = {}
                    for req_col in required_cols:
                        for col in cut_x_simple.columns:
                            if req_col.lower() in col.lower():
                                col_mapping[col] = req_col
                                break
                    
                    if col_mapping:
                        cut_x_simple = cut_x_simple.rename(columns=col_mapping)
                
                # Verify minimum size
                min_size = cut_x_simple['CUT SIDE (mm)'].min() if 'CUT SIDE (mm)' in cut_x_simple.columns else None
                if min_size is not None:
                    print(f"Minimum Cut X size: {min_size}mm")
                    
                # Add Remaining Size column
                cut_x_simple_with_remaining = cut_x_simple.copy()
                cut_x_simple_with_remaining['Remaining Size (mm)'] = tile_width - cut_x_simple_with_remaining['CUT SIDE (mm)']
                
                # Rename columns for clarity
                cut_x_simple_with_remaining.rename(columns={
                    'APPARTMENT NUMBER': 'Apartment',
                    'CUT SIDE (mm)': 'Cut Size (mm)',
                    'LOCATION': 'Location',
                    'COUNT': 'Count'
                }, inplace=True)
                
                # Sort by cut size (small to big)
                cut_x_simple_with_remaining = cut_x_simple_with_remaining.sort_values('Cut Size (mm)')
                
                # Display the data
                print(f"Total Cut X items: {len(cut_x_simple_with_remaining)}")
            else:
                print("No Cut X data available.")
                cut_x_simple_with_remaining = pd.DataFrame()
            
            # --- Process Cut Y data ---
            print("\n📋 Cut Y Data (with small tiles < {size_threshold}mm excluded):")
            
            # Handle different data formats
            cut_y_simple = export_info.get('cut_y_simple', [])
            if isinstance(cut_y_simple, list):
                cut_y_simple = pd.DataFrame(cut_y_simple)
                
            if not isinstance(cut_y_simple, pd.DataFrame):
                cut_y_simple = pd.DataFrame()
            
            if not cut_y_simple.empty:
                # Verify column names exist
                required_cols = ['APPARTMENT NUMBER', 'CUT SIDE (mm)', 'LOCATION', 'COUNT']
                if not all(col in cut_y_simple.columns for col in required_cols):
                    # Try to find similar column names if exact ones don't exist
                    col_mapping = {}
                    for req_col in required_cols:
                        for col in cut_y_simple.columns:
                            if req_col.lower() in col.lower():
                                col_mapping[col] = req_col
                                break
                    
                    if col_mapping:
                        cut_y_simple = cut_y_simple.rename(columns=col_mapping)
                
                # Verify minimum size
                min_size = cut_y_simple['CUT SIDE (mm)'].min() if 'CUT SIDE (mm)' in cut_y_simple.columns else None
                if min_size is not None:
                    print(f"Minimum Cut Y size: {min_size}mm")
                    
                # Add Remaining Size column
                cut_y_simple_with_remaining = cut_y_simple.copy()
                cut_y_simple_with_remaining['Remaining Size (mm)'] = tile_height - cut_y_simple_with_remaining['CUT SIDE (mm)']
                
                # Rename columns for clarity
                cut_y_simple_with_remaining.rename(columns={
                    'APPARTMENT NUMBER': 'Apartment',
                    'CUT SIDE (mm)': 'Cut Size (mm)',
                    'LOCATION': 'Location',
                    'COUNT': 'Count'
                }, inplace=True)
                
                # Sort by cut size (small to big)
                cut_y_simple_with_remaining = cut_y_simple_with_remaining.sort_values('Cut Size (mm)')
                
                # Display the data
                print(f"Total Cut Y items: {len(cut_y_simple_with_remaining)}")
            else:
                print("No Cut Y data available.")
                cut_y_simple_with_remaining = pd.DataFrame()
            
            # Store the final dataframes for future use
            cut_data_for_matching = {
                'cut_x_data': cut_x_simple_with_remaining,
                'cut_y_data': cut_y_simple_with_remaining,
                'has_pattern': has_pattern,
                'tile_width': tile_width,
                'tile_height': tile_height
            }
            
        else:
            # --- Process All Cut data ---
            print("\n📋 All Cut Data (with small tiles < {size_threshold}mm excluded):")
            
            # Handle different data formats
            all_cut_simple = export_info.get('all_cut_simple', [])
            if isinstance(all_cut_simple, list):
                all_cut_simple = pd.DataFrame(all_cut_simple)
                
            if not isinstance(all_cut_simple, pd.DataFrame):
                all_cut_simple = pd.DataFrame()
            
            if not all_cut_simple.empty:
                # Verify column names exist
                required_cols = ['APPARTMENT NUMBER', 'CUT DIMENSION (mm)', 'LOCATION', 'COUNT']
                if not all(col in all_cut_simple.columns for col in required_cols):
                    # Try to find similar column names if exact ones don't exist
                    col_mapping = {}
                    for req_col in required_cols:
                        for col in all_cut_simple.columns:
                            if req_col.lower() in col.lower():
                                col_mapping[col] = req_col
                                break
                    
                    if col_mapping:
                        all_cut_simple = all_cut_simple.rename(columns=col_mapping)
                
                # Verify minimum size
                min_size = all_cut_simple['CUT DIMENSION (mm)'].min() if 'CUT DIMENSION (mm)' in all_cut_simple.columns else None
                if min_size is not None:
                    print(f"Minimum cut dimension: {min_size}mm")
                    
                # Add Remaining Size column
                all_cut_simple_with_remaining = all_cut_simple.copy()
                min_dimension = min(tile_width, tile_height)
                all_cut_simple_with_remaining['Remaining Size (mm)'] = min_dimension - all_cut_simple_with_remaining['CUT DIMENSION (mm)']
                
                # Rename columns for clarity
                all_cut_simple_with_remaining.rename(columns={
                    'APPARTMENT NUMBER': 'Apartment',
                    'CUT DIMENSION (mm)': 'Cut Size (mm)',
                    'LOCATION': 'Location',
                    'COUNT': 'Count'
                }, inplace=True)
                
                # Sort by cut size (small to big)
                all_cut_simple_with_remaining = all_cut_simple_with_remaining.sort_values('Cut Size (mm)')
                
                # Display the data
                print(f"Total All Cut items: {len(all_cut_simple_with_remaining)}")
            else:
                print("No All Cut data available.")
                all_cut_simple_with_remaining = pd.DataFrame()
            
            # Store the final dataframes for future use
            cut_data_for_matching = {
                'all_cut_data': all_cut_simple_with_remaining,
                'has_pattern': has_pattern,
                'tile_width': tile_width,
                'tile_height': tile_height
            }
        
        # Check if inventory data is available from previous step
        inventory_for_matching = None
        if 'inventory_for_matching' in globals():
            print("\n📦 Found inventory data from previous step.")
            inventory_for_matching = globals()['inventory_for_matching']
        
        # Process inventory if it exists
        if inventory_for_matching:
            # Process inventory data based on pattern mode
            if has_pattern:
                # --- Process Cut X Inventory ---
                print("\n📋 Cut X Inventory:")
                
                # Get Cut X inventory data
                cut_x_inventory = inventory_for_matching.get('cut_x_inventory', pd.DataFrame())
                if not isinstance(cut_x_inventory, pd.DataFrame):
                    if isinstance(cut_x_inventory, list):
                        cut_x_inventory = pd.DataFrame(cut_x_inventory)
                    else:
                        cut_x_inventory = pd.DataFrame()
                
                if cut_x_inventory.empty and 'cut_x_data' in inventory_for_matching:
                    cut_x_inventory = inventory_for_matching['cut_x_data']
                    if not isinstance(cut_x_inventory, pd.DataFrame):
                        if isinstance(cut_x_inventory, list):
                            cut_x_inventory = pd.DataFrame(cut_x_inventory)
                        else:
                            cut_x_inventory = pd.DataFrame()
                
                # --- Process Cut Y Inventory ---
                print("\n📋 Cut Y Inventory:")
                
                # Get Cut Y inventory data
                cut_y_inventory = inventory_for_matching.get('cut_y_inventory', pd.DataFrame())
                if not isinstance(cut_y_inventory, pd.DataFrame):
                    if isinstance(cut_y_inventory, list):
                        cut_y_inventory = pd.DataFrame(cut_y_inventory)
                    else:
                        cut_y_inventory = pd.DataFrame()
                
                if cut_y_inventory.empty and 'cut_y_data' in inventory_for_matching:
                    cut_y_inventory = inventory_for_matching['cut_y_data']
                    if not isinstance(cut_y_inventory, pd.DataFrame):
                        if isinstance(cut_y_inventory, list):
                            cut_y_inventory = pd.DataFrame(cut_y_inventory)
                        else:
                            cut_y_inventory = pd.DataFrame()
                
                # Add inventory data to cut_data_for_matching
                if not cut_x_inventory.empty:
                    cut_data_for_matching['cut_x_inventory'] = cut_x_inventory
                    total_x = cut_x_inventory['Count'].sum() if 'Count' in cut_x_inventory.columns else len(cut_x_inventory)
                    print(f"Total Cut X pieces in inventory: {total_x}")
                else:
                    print("\n📋 No Cut X inventory available.")
                
                if not cut_y_inventory.empty:
                    cut_data_for_matching['cut_y_inventory'] = cut_y_inventory
                    total_y = cut_y_inventory['Count'].sum() if 'Count' in cut_y_inventory.columns else len(cut_y_inventory)
                    print(f"Total Cut Y pieces in inventory: {total_y}")
                else:
                    print("\n📋 No Cut Y inventory available.")
            else:
                # --- Process All Cut Inventory ---
                print("\n📋 All Cut Inventory:")
                
                # Get All Cut inventory data
                all_cut_inventory = inventory_for_matching.get('all_cut_inventory', pd.DataFrame())
                if not isinstance(all_cut_inventory, pd.DataFrame):
                    if isinstance(all_cut_inventory, list):
                        all_cut_inventory = pd.DataFrame(all_cut_inventory)
                    else:
                        all_cut_inventory = pd.DataFrame()
                
                if all_cut_inventory.empty and 'all_cut_data' in inventory_for_matching:
                    all_cut_inventory = inventory_for_matching['all_cut_data']
                    if not isinstance(all_cut_inventory, pd.DataFrame):
                        if isinstance(all_cut_inventory, list):
                            all_cut_inventory = pd.DataFrame(all_cut_inventory)
                        else:
                            all_cut_inventory = pd.DataFrame()
                
                # Add inventory data to cut_data_for_matching
                if not all_cut_inventory.empty:
                    cut_data_for_matching['all_cut_inventory'] = all_cut_inventory
                    total_all = all_cut_inventory['Count'].sum() if 'Count' in all_cut_inventory.columns else len(all_cut_inventory)
                    print(f"Total All Cut pieces in inventory: {total_all}")
                else:
                    print("\n📋 No All Cut inventory available.")
        else:
            print("\n📋 No inventory data found from previous step.")
            print("   If you want to match with inventory in the next step, please run the inventory setup step first.")
        
        print("\n✅ Step 8A complete! Data is prepared and ready for matching.")
        return cut_data_for_matching
    
    def split_cut_pieces_by_half(self, cut_data_for_matching):
        """
        Split cut pieces by half-tile threshold - Modified to match Colab Step 8B
        
        Parameters:
        - cut_data_for_matching: Dict with cut data for matching
        
        Returns:
        - Dict with cut pieces split by half threshold
        """
        # Get pattern mode and tile dimensions
        has_pattern = cut_data_for_matching.get('has_pattern', True)
        tile_width = cut_data_for_matching.get('tile_width', 600)
        tile_height = cut_data_for_matching.get('tile_height', 600)
        
        # Calculate half tile threshold
        half_threshold = min(tile_width/2, tile_height/2)
        
        if has_pattern:
            # Process Cut X Tiles
            cut_x_data = cut_data_for_matching.get('cut_x_data', pd.DataFrame())
            
            if not cut_x_data.empty:
                # Split into less than half and more than half
                x_less_than_half = cut_x_data[cut_x_data['Cut Size (mm)'] < half_threshold].copy()
                x_more_than_half = cut_x_data[cut_x_data['Cut Size (mm)'] >= half_threshold].copy()
                
                # Ensure sorting
                x_less_than_half = x_less_than_half.sort_values('Cut Size (mm)')
                x_more_than_half = x_more_than_half.sort_values('Cut Size (mm)')
            else:
                x_less_than_half = pd.DataFrame()
                x_more_than_half = pd.DataFrame()
            
            # Process Cut Y Tiles
            cut_y_data = cut_data_for_matching.get('cut_y_data', pd.DataFrame())
            
            if not cut_y_data.empty:
                # Split into less than half and more than half
                y_less_than_half = cut_y_data[cut_y_data['Cut Size (mm)'] < half_threshold].copy()
                y_more_than_half = cut_y_data[cut_y_data['Cut Size (mm)'] >= half_threshold].copy()
                
                # Ensure sorting
                y_less_than_half = y_less_than_half.sort_values('Cut Size (mm)')
                y_more_than_half = y_more_than_half.sort_values('Cut Size (mm)')
            else:
                y_less_than_half = pd.DataFrame()
                y_more_than_half = pd.DataFrame()
            
            # Store cut pieces for next step
            cut_pieces_by_half = {
                'x_less_than_half': x_less_than_half,
                'x_more_than_half': x_more_than_half,
                'y_less_than_half': y_less_than_half,
                'y_more_than_half': y_more_than_half,
                'half_threshold': half_threshold,
                'tile_width': tile_width,
                'tile_height': tile_height,
                'has_pattern': has_pattern
            }
        else:
            # Process All Cut Tiles
            all_cut_data = cut_data_for_matching.get('all_cut_data', pd.DataFrame())
            
            if not all_cut_data.empty:
                # Split into less than half and more than half
                all_less_than_half = all_cut_data[all_cut_data['Cut Size (mm)'] < half_threshold].copy()
                all_more_than_half = all_cut_data[all_cut_data['Cut Size (mm)'] >= half_threshold].copy()
                
                # Ensure sorting
                all_less_than_half = all_less_than_half.sort_values('Cut Size (mm)')
                all_more_than_half = all_more_than_half.sort_values('Cut Size (mm)')
            else:
                all_less_than_half = pd.DataFrame()
                all_more_than_half = pd.DataFrame()
            
            # Store cut pieces for next step
            cut_pieces_by_half = {
                'all_less_than_half': all_less_than_half,
                'all_more_than_half': all_more_than_half,
                'half_threshold': half_threshold,
                'tile_width': tile_width,
                'tile_height': tile_height,
                'has_pattern': has_pattern
            }
        
        # Process inventory data if provided
        inventory_data = cut_data_for_matching.get('inventory_data')
        if inventory_data:
            # Split inventory data by half threshold similar to cut pieces
            self._split_inventory_data(cut_pieces_by_half, inventory_data)
        
        return cut_pieces_by_half
    
    def _split_inventory_data(self, cut_pieces_by_half, inventory_data):
        """
        Split inventory data by half threshold
        
        Parameters:
        - cut_pieces_by_half: Dict to store split inventory data
        - inventory_data: Dict with inventory data
        """
        has_pattern = cut_pieces_by_half.get('has_pattern', True)
        tile_width = cut_pieces_by_half.get('tile_width', 600)
        tile_height = cut_pieces_by_half.get('tile_height', 600)
        half_threshold = cut_pieces_by_half.get('half_threshold')
        
        if has_pattern:
            # Process Cut X Inventory
            cut_x_inventory = inventory_data.get('cut_x_inventory', pd.DataFrame())
            
            if not cut_x_inventory.empty:
                # Split based on remaining size
                x_inv_less_than_half = cut_x_inventory[cut_x_inventory['Remaining Size (mm)'] > (tile_width - half_threshold)].copy()
                x_inv_more_than_half = cut_x_inventory[cut_x_inventory['Remaining Size (mm)'] <= (tile_width - half_threshold)].copy()
                
                # Sort (largest remaining pieces first)
                x_inv_less_than_half = x_inv_less_than_half.sort_values('Remaining Size (mm)', ascending=False)
                x_inv_more_than_half = x_inv_more_than_half.sort_values('Remaining Size (mm)', ascending=False)
                
                # Add to cut_pieces_by_half
                cut_pieces_by_half['x_inv_less_than_half'] = x_inv_less_than_half
                cut_pieces_by_half['x_inv_more_than_half'] = x_inv_more_than_half
            
            # Process Cut Y Inventory
            cut_y_inventory = inventory_data.get('cut_y_inventory', pd.DataFrame())
            
            if not cut_y_inventory.empty:
                # Split based on remaining size
                y_inv_less_than_half = cut_y_inventory[cut_y_inventory['Remaining Size (mm)'] > (tile_height - half_threshold)].copy()
                y_inv_more_than_half = cut_y_inventory[cut_y_inventory['Remaining Size (mm)'] <= (tile_height - half_threshold)].copy()
                
                # Sort (largest remaining pieces first)
                y_inv_less_than_half = y_inv_less_than_half.sort_values('Remaining Size (mm)', ascending=False)
                y_inv_more_than_half = y_inv_more_than_half.sort_values('Remaining Size (mm)', ascending=False)
                
                # Add to cut_pieces_by_half
                cut_pieces_by_half['y_inv_less_than_half'] = y_inv_less_than_half
                cut_pieces_by_half['y_inv_more_than_half'] = y_inv_more_than_half
            
        else:
            # Process All Cut Inventory
            all_cut_inventory = inventory_data.get('all_cut_inventory', pd.DataFrame())
            
            if not all_cut_inventory.empty:
                # Split based on remaining size
                min_dimension = min(tile_width, tile_height)
                all_inv_less_than_half = all_cut_inventory[all_cut_inventory['Remaining Size (mm)'] > (min_dimension - half_threshold)].copy()
                all_inv_more_than_half = all_cut_inventory[all_cut_inventory['Remaining Size (mm)'] <= (min_dimension - half_threshold)].copy()
                
                # Sort (largest remaining pieces first)
                all_inv_less_than_half = all_inv_less_than_half.sort_values('Remaining Size (mm)', ascending=False)
                all_inv_more_than_half = all_inv_more_than_half.sort_values('Remaining Size (mm)', ascending=False)
                
                # Add to cut_pieces_by_half
                cut_pieces_by_half['all_inv_less_than_half'] = all_inv_less_than_half
                cut_pieces_by_half['all_inv_more_than_half'] = all_inv_more_than_half
    
    def expand_dataframe(self, df):
        """
        Expand DataFrame rows based on Count column
        
        Parameters:
        - df: DataFrame to expand
        
        Returns:
        - List of dictionaries representing expanded rows
        """
        if isinstance(df, list):
            return df
            
        if df.empty:
            return []
            
        expanded = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            # Make sure Count is interpreted as an integer
            count = int(float(row_dict.get('Count', 1))) if 'Count' in row_dict else 1
            
            # Remove Count from the dictionary
            if 'Count' in row_dict:
                del row_dict['Count']
            
            # Add this row count times
            for _ in range(count):
                expanded.append(row_dict.copy())
        
        return expanded
    
    def match_with_inventory(self, cut_pieces, inventory_pieces, tolerance_ranges, direction="X"):
        """
        Match cut pieces with inventory using progressive tolerance
        
        Parameters:
        - cut_pieces: List of dictionaries representing cut pieces
        - inventory_pieces: List of dictionaries representing inventory pieces
        - tolerance_ranges: List of tolerance values to use
        - direction: Direction of the cuts ('X', 'Y', or 'All')
        
        Returns:
        - Tuple of (matches, unmatched_cut, unmatched_inv)
        """
        matches = []
        inv_counter = 1  # Counter for inventory matches
        
        # Check for empty inputs
        if not cut_pieces or not inventory_pieces:
            print(f"Empty input: cut_pieces={len(cut_pieces)}, inventory_pieces={len(inventory_pieces)}")
            return matches, cut_pieces, inventory_pieces
        
        # Make copies to track matches
        cut_copy = [item.copy() for item in cut_pieces]
        inv_copy = [item.copy() for item in inventory_pieces]
        
        # Add matched flag
        for item in cut_copy:
            item['Matched'] = False
        for item in inv_copy:
            item['Matched'] = False
        
        # Debug info
        print(f"Matching {len(cut_copy)} cut pieces with {len(inv_copy)} inventory pieces")
        print(f"Sample cut piece: {cut_copy[0] if cut_copy else 'None'}")
        print(f"Sample inventory piece: {inv_copy[0] if inv_copy else 'None'}")
            
        # Process each tolerance range
        for tolerance in tolerance_ranges:
            match_count = 0
            print(f"   Processing inventory matches with tolerance <= {tolerance}mm...")
            
            # For each cut piece
            for i, cut in enumerate(cut_copy):
                if cut['Matched']:
                    continue
                
                # Get cut size - make sure to handle different column names
                cut_size = cut.get('Cut Size (mm)')
                if cut_size is None:
                    cut_size = cut.get('Cut Size')
                if cut_size is None and 'cut_size' in cut:
                    cut_size = cut['cut_size']
                
                if cut_size is None:
                    print(f"Warning: Could not find cut size in {cut.keys()}")
                    continue
                
                cut_apt = cut.get('Apartment', '')
                cut_loc = cut.get('Location', '')
                
                # Get remaining size - handle different column names
                cut_remain = cut.get('Remaining Size (mm)')
                if cut_remain is None:
                    cut_remain = cut.get('Remaining Size')
                if cut_remain is None and 'remaining_size' in cut:
                    cut_remain = cut['remaining_size']
                
                if cut_remain is None:
                    print(f"Warning: Could not find remaining size in {cut.keys()}")
                    continue
                
                best_match_idx = None
                best_waste = float('inf')
                
                # For each inventory piece
                for j, inv in enumerate(inv_copy):
                    if inv['Matched']:
                        continue
                    
                    # Get inventory remaining size - handle different column names
                    inv_size = inv.get('Remaining Size (mm)')
                    if inv_size is None:
                        inv_size = inv.get('Size (mm)')
                    if inv_size is None:
                        inv_size = inv.get('Remaining Size')
                    if inv_size is None and 'remaining_size' in inv:
                        inv_size = inv['remaining_size']
                        
                    if inv_size is None:
                        print(f"Warning: Could not find inventory size in {inv.keys()}")
                        continue
                    
                    inv_loc = inv.get('Location', '')
                    
                    # Check if inventory size is greater than or equal to cut size
                    if inv_size >= cut_size:
                        # Calculate waste
                        waste = inv_size - cut_size
                        
                        # Only consider matches within current tolerance
                        if waste <= tolerance and waste < best_waste:
                            best_waste = waste
                            best_match_idx = j
                
                if best_match_idx is not None:
                    # Found a match!
                    inv = inv_copy[best_match_idx]
                    
                    # Get inventory size and location again (for clarity)
                    inv_size = inv.get('Remaining Size (mm)')
                    if inv_size is None:
                        inv_size = inv.get('Size (mm)')
                    if inv_size is None:
                        inv_size = inv.get('Remaining Size')
                    if inv_size is None and 'remaining_size' in inv:
                        inv_size = inv['remaining_size']
                    
                    inv_loc = inv.get('Location', '')
                    
                    # Set match ID based on direction
                    if direction == "X":
                        match_id = f"IX{inv_counter}"
                    elif direction == "Y":
                        match_id = f"IY{inv_counter}"
                    else:  # All cuts (no pattern)
                        match_id = f"I{inv_counter}"
                    
                    inv_counter += 1
                    match_count += 1
                    
                    # Add to matches
                    matches.append({
                        'Match ID': match_id,
                        'Apartment': cut_apt,
                        'Location': cut_loc,
                        'Cut Size (mm)': cut_size,
                        'Cut Remain (mm)': cut_remain,
                        'Inventory Location': inv_loc,
                        'Inventory Size (mm)': inv_size,
                        'Waste': best_waste,
                        'Tolerance Range': f"<={tolerance}mm"
                    })
                    
                    # Mark as matched
                    cut_copy[i]['Matched'] = True
                    inv_copy[best_match_idx]['Matched'] = True
            
            print(f"      Found {match_count} inventory matches in tolerance range <= {tolerance}mm")
        
        # Return matches and unmatched pieces
        unmatched_cut = [item for item in cut_copy if not item.get('Matched', False)]
        unmatched_inv = [item for item in inv_copy if not item.get('Matched', False)]
        
        print(f"Total inventory matches: {len(matches)}")
        print(f"Remaining unmatched cut pieces: {len(unmatched_cut)}")
        print(f"Remaining unmatched inventory pieces: {len(unmatched_inv)}")
        
        return matches, unmatched_cut, unmatched_inv
    
    def match_with_progressive_tolerance(self, less_pieces, more_pieces, tolerance_ranges, direction="X"):
        """
        Match pieces with progressive tolerance
        
        Parameters:
        - less_pieces: List of dictionaries representing less than half pieces
        - more_pieces: List of dictionaries representing more than half pieces
        - tolerance_ranges: List of tolerance values to use
        - direction: Direction of the cuts ('X', 'Y', or 'All')
        
        Returns:
        - Tuple of (matches, unmatched_less, unmatched_more)
        """
        matches = []
        same_apt_counter = 1  # Counter for same apartment matches
        diff_apt_counter = 1  # Counter for different apartment matches
        
        # Check for empty inputs
        if not less_pieces or not more_pieces:
            return matches, less_pieces, more_pieces
        
        # Make copies to track matches
        less_copy = [item.copy() for item in less_pieces]
        more_copy = [item.copy() for item in more_pieces]
        
        # Add matched flag
        for item in less_copy:
            item['Matched'] = False
        for item in more_copy:
            item['Matched'] = False
        
        # Process each tolerance range
        for tolerance in tolerance_ranges:
            print(f"   Processing apartment matches with tolerance <= {tolerance}mm...")
            
            matches_in_range = 0
            
            # For each less than half piece
            for i, less in enumerate(less_copy):
                if less.get('Matched', False):
                    continue
                
                less_size = less.get('Cut Size (mm)', 0)
                less_apt = less.get('Apartment', '')
                less_loc = less.get('Location', '')
                less_remain = less.get('Remaining Size (mm)', 0)
                
                best_match_idx = None
                best_waste = float('inf')
                
                # For each more than half piece
                for j, more in enumerate(more_copy):
                    if more.get('Matched', False):
                        continue
                    
                    more_size = more.get('Cut Size (mm)', 0)
                    more_apt = more.get('Apartment', '')
                    more_loc = more.get('Location', '')
                    
                    # Check if remaining size is greater than cut size
                    if less_remain > more_size:
                        # Calculate waste
                        waste = less_remain - more_size
                        
                        # Only consider matches within current tolerance
                        if waste <= tolerance and waste < best_waste:
                            best_waste = waste
                            best_match_idx = j
                
                if best_match_idx is not None:
                    # Found a match!
                    more = more_copy[best_match_idx]
                    more_size = more.get('Cut Size (mm)', 0)
                    more_apt = more.get('Apartment', '')
                    more_loc = more.get('Location', '')
                    
                    # Determine match type
                    is_same_apt = less_apt == more_apt
                    
                    # Set match ID based on apartment match and direction
                    if direction == "X":
                        if is_same_apt:
                            match_id = f"X{same_apt_counter}"
                            same_apt_counter += 1
                        else:
                            match_id = f"OX{diff_apt_counter}"
                            diff_apt_counter += 1
                    elif direction == "Y":
                        if is_same_apt:
                            match_id = f"Y{same_apt_counter}"
                            same_apt_counter += 1
                        else:
                            match_id = f"OY{diff_apt_counter}"
                            diff_apt_counter += 1
                    else:  # All cuts (no pattern)
                        if is_same_apt:
                            match_id = f"XY{same_apt_counter}"
                            same_apt_counter += 1
                        else:
                            match_id = f"O{diff_apt_counter}"
                            diff_apt_counter += 1
                    
                    matches_in_range += 1
                    
                    # Add to matches
                    matches.append({
                        'Match ID': match_id,
                        'Small Piece Apt': less_apt,
                        'Small Piece Loc': less_loc,
                        'Small Piece Size': less_size,
                        'Small Piece Remain': less_remain,
                        'Large Piece Apt': more_apt,
                        'Large Piece Loc': more_loc,
                        'Large Piece Size': more_size,
                        'Waste': best_waste,
                        'Tolerance Range': f"<={tolerance}mm",
                        'Same Apartment': is_same_apt
                    })
                    
                    # Mark as matched
                    less_copy[i]['Matched'] = True
                    more_copy[best_match_idx]['Matched'] = True
                
                print(f"      Found {matches_in_range} apartment matches in tolerance range <= {tolerance}mm")
        
        # Return matches and unmatched pieces
        unmatched_less = [item for item in less_copy if not item.get('Matched', False)]
        unmatched_more = [item for item in more_copy if not item.get('Matched', False)]
        
        return matches, unmatched_less, unmatched_more
    
    def process_inventory_for_matching(self, inventory_data):
        """
        Process inventory data for matching
        
        Parameters:
        - inventory_data: Dict with inventory data
        
        Returns:
        - Dict with processed inventory data for matching
        """
        if not inventory_data:
            return None
            
        has_pattern = inventory_data.get('has_pattern', True)
        
        # Create dictionary to store processed data
        processed_inventory = {'has_pattern': has_pattern}
        
        if has_pattern:
            # Process X direction inventory
            x_inventory = inventory_data.get('cut_x_inventory', None)
            
            # Try different possible keys if the main key is not found
            if x_inventory is None:
                x_inventory = inventory_data.get('x_inv_less_than_half', None)
            if x_inventory is None:
                x_inventory = inventory_data.get('cut_x_data', None)
            
            # Convert to DataFrame if it's a list
            if isinstance(x_inventory, list):
                x_inventory = pd.DataFrame(x_inventory) if x_inventory else pd.DataFrame()
            
            if not isinstance(x_inventory, pd.DataFrame):
                x_inventory = pd.DataFrame()
            
            if not x_inventory.empty:
                # Make sure all required columns exist
                if 'Remaining Size (mm)' not in x_inventory.columns:
                    if 'Size (mm)' in x_inventory.columns:
                        x_inventory['Remaining Size (mm)'] = x_inventory['Size (mm)']
                    
                if 'Location' not in x_inventory.columns:
                    x_inventory['Location'] = 'INV'
                    
                if 'Count' not in x_inventory.columns:
                    x_inventory['Count'] = 1
                    
                processed_inventory['cut_x_inventory'] = x_inventory
            
            # Process Y direction inventory - similar logic as X
            y_inventory = inventory_data.get('cut_y_inventory', None)
            
            # Try different possible keys if the main key is not found
            if y_inventory is None:
                y_inventory = inventory_data.get('y_inv_less_than_half', None)
            if y_inventory is None:
                y_inventory = inventory_data.get('cut_y_data', None)
            
            # Convert to DataFrame if it's a list
            if isinstance(y_inventory, list):
                y_inventory = pd.DataFrame(y_inventory) if y_inventory else pd.DataFrame()
            
            if not isinstance(y_inventory, pd.DataFrame):
                y_inventory = pd.DataFrame()
            
            if not y_inventory.empty:
                # Make sure all required columns exist
                if 'Remaining Size (mm)' not in y_inventory.columns:
                    if 'Size (mm)' in y_inventory.columns:
                        y_inventory['Remaining Size (mm)'] = y_inventory['Size (mm)']
                        
                if 'Location' not in y_inventory.columns:
                    y_inventory['Location'] = 'INV'
                    
                if 'Count' not in y_inventory.columns:
                    y_inventory['Count'] = 1
                    
                processed_inventory['cut_y_inventory'] = y_inventory
        else:
            # Process non-pattern inventory
            all_inventory = inventory_data.get('all_cut_inventory', None)
            
            # Try different possible keys if the main key is not found
            if all_inventory is None:
                all_inventory = inventory_data.get('all_inv_less_than_half', None)
            if all_inventory is None:
                all_inventory = inventory_data.get('all_cut_data', None)
            
            # Convert to DataFrame if it's a list
            if isinstance(all_inventory, list):
                all_inventory = pd.DataFrame(all_inventory) if all_inventory else pd.DataFrame()
            
            if not isinstance(all_inventory, pd.DataFrame):
                all_inventory = pd.DataFrame()
            
            if not all_inventory.empty:
                # Make sure all required columns exist
                if 'Remaining Size (mm)' not in all_inventory.columns:
                    if 'Size (mm)' in all_inventory.columns:
                        all_inventory['Remaining Size (mm)'] = all_inventory['Size (mm)']
                        
                if 'Location' not in all_inventory.columns:
                    all_inventory['Location'] = 'INV'
                    
                if 'Count' not in all_inventory.columns:
                    all_inventory['Count'] = 1
                    
                processed_inventory['all_cut_inventory'] = all_inventory
        
        return processed_inventory

    
    def match_cut_pieces(self, cut_pieces_by_half, tolerance_ranges, inventory_data=None):
        """
        Match cut pieces with inventory and progressive tolerance - Modified to match Colab Step 8C
        
        Parameters:
        - cut_pieces_by_half: Dict with cut pieces split by half threshold
        - tolerance_ranges: List of tolerance values to use
        - inventory_data: Dict with inventory data (optional)
        
        Returns:
        - Dict with matching results
        """
        print(f"Pattern mode: {'With Pattern (separate X/Y cuts)' if cut_pieces_by_half.get('has_pattern', True) else 'No Pattern (all cuts)'}")
        print(f"Standard tile size: {cut_pieces_by_half.get('tile_width', 600)}mm x {cut_pieces_by_half.get('tile_height', 600)}mm")
        print(f"Half-tile threshold: {cut_pieces_by_half.get('half_threshold', 300)}mm")
        print(f"Using progressive tolerance ranges (mm): {tolerance_ranges}")
        max_tolerance = min(cut_pieces_by_half.get('tile_width', 600), cut_pieces_by_half.get('tile_height', 600)) / 4
        print(f"Maximum tolerance: {max_tolerance:.1f}mm (1/4 of smallest dimension)")
        
        # Get pattern mode and other info
        has_pattern = cut_pieces_by_half.get('has_pattern', True)
        
        # Initialize all variables that will be used later
        x_inv_matches = []
        y_inv_matches = []
        x_apt_matches = []
        y_apt_matches = []
        all_inv_matches = []
        all_apt_matches = []
        x_inv_matches_df = pd.DataFrame()
        y_inv_matches_df = pd.DataFrame()
        x_apt_matches_df = pd.DataFrame()
        y_apt_matches_df = pd.DataFrame()
        all_inv_matches_df = pd.DataFrame()
        all_apt_matches_df = pd.DataFrame()
        x_unmatched_less_agg = pd.DataFrame()
        x_unmatched_more_agg = pd.DataFrame()
        y_unmatched_less_agg = pd.DataFrame()
        y_unmatched_more_agg = pd.DataFrame()
        all_unmatched_less_agg = pd.DataFrame()
        all_unmatched_more_agg = pd.DataFrame()
        
        # Initialize counters
        within_apartment_matches = 0
        cross_apartment_matches = 0
        inventory_matches = 0
        total_savings = 0
        
        # Process inventory data if provided
        has_inventory = False
        inventory_data_processed = None

        if inventory_data is not None:
            has_inventory = True
            print(f"\n📦 Inventory data provided. Processing for matching...")
            inventory_data_processed = self.process_inventory_for_matching(inventory_data)
            
            if inventory_data_processed:
                print(f"Processed inventory data with keys: {list(inventory_data_processed.keys())}")
                for key, value in inventory_data_processed.items():
                    if key != 'has_pattern' and isinstance(value, pd.DataFrame):
                        print(f"   {key}: {len(value)} rows")
                        # Display first row for debugging
                        if not value.empty:
                            print(f"   First row sample: {value.iloc[0].to_dict()}")
        
        # Create dict to store results
        match_results = {
            'has_pattern': has_pattern,
            'has_inventory': has_inventory,
            'tolerance_ranges': tolerance_ranges
        }
        
        # Process matching based on pattern mode
        if has_pattern:
            # =============== PROCESS X DIRECTION ===============
            print("\n📋 Processing Cut X Tiles:")
            
            # Get X direction pieces
            x_less_than_half = cut_pieces_by_half.get('x_less_than_half', pd.DataFrame())
            x_more_than_half = cut_pieces_by_half.get('x_more_than_half', pd.DataFrame())
            
            if isinstance(x_less_than_half, list):
                x_less_than_half = pd.DataFrame(x_less_than_half) if x_less_than_half else pd.DataFrame()
            if isinstance(x_more_than_half, list):
                x_more_than_half = pd.DataFrame(x_more_than_half) if x_more_than_half else pd.DataFrame()
            
            print(f"X less than half: {len(x_less_than_half)} rows")
            print(f"X more than half: {len(x_more_than_half)} rows")
            
            # Track inventory matches and remaining pieces
            x_less_after_inv = []
            
            # First, try inventory matching if available
            if has_inventory and not x_less_than_half.empty and inventory_data_processed:
                print("\n🔄 Matching Cut X Less than Half with Inventory...")
                
                # Get X direction inventory data - check all possible keys
                x_inv_less = None
                if 'cut_x_inventory' in inventory_data_processed and not inventory_data_processed['cut_x_inventory'].empty:
                    x_inv_less = inventory_data_processed['cut_x_inventory']
                    print(f"Found cut_x_inventory with {len(x_inv_less)} rows")
                elif 'x_inv_less_than_half' in cut_pieces_by_half and not isinstance(cut_pieces_by_half['x_inv_less_than_half'], list) and not cut_pieces_by_half['x_inv_less_than_half'].empty:
                    x_inv_less = cut_pieces_by_half['x_inv_less_than_half']
                    print(f"Found x_inv_less_than_half with {len(x_inv_less)} rows")
                elif isinstance(inventory_data, dict) and 'cut_x_inventory' in inventory_data and not inventory_data['cut_x_inventory'].empty:
                    x_inv_less = inventory_data['cut_x_inventory']
                    print(f"Found cut_x_inventory directly in inventory_data with {len(x_inv_less)} rows")
                
                if x_inv_less is not None and not x_inv_less.empty:
                    # Expand DataFrames
                    x_less_expanded = self.expand_dataframe(x_less_than_half)
                    x_inv_expanded = self.expand_dataframe(x_inv_less)
                    
                    print(f"Expanded X less than half: {len(x_less_expanded)} individual tiles")
                    print(f"Expanded X inventory: {len(x_inv_expanded)} individual pieces")
                    
                    # Match with inventory
                    x_inv_matches, x_less_after_inv, _ = self.match_with_inventory(
                        x_less_expanded, x_inv_expanded, tolerance_ranges, "X")
                    print(f"Found {len(x_inv_matches)} X inventory matches")
                    
                    # Update counter
                    inventory_matches += len(x_inv_matches)
                    
                    # Update savings
                    for match in x_inv_matches:
                        total_savings += match.get('Cut Size (mm)', 0)
                else:
                    # No inventory data, use all less pieces for apartment matching
                    x_less_after_inv = self.expand_dataframe(x_less_than_half)
                    print("No suitable X inventory data found, skipping inventory matching")
            else:
                # No inventory, use all less pieces for apartment matching
                if not x_less_than_half.empty:
                    x_less_after_inv = self.expand_dataframe(x_less_than_half)
                    reason = "No inventory provided" if not has_inventory else "Empty less_than_half" if x_less_than_half.empty else "No processed inventory data"
                    print(f"Skipping X inventory matching ({reason})")
                else:
                    x_less_after_inv = []
                    print("No X less-than-half pieces to match")
            
            # Now do apartment-to-apartment matching for remaining pieces
            x_unmatched_less = []
            x_unmatched_more = []
            
            if len(x_less_after_inv) > 0 and not x_more_than_half.empty:
                print(f"\n🔄 Matching remaining Cut X pieces between apartments...")
                
                # Expand more than half pieces
                x_more_expanded = self.expand_dataframe(x_more_than_half)
                
                print(f"Remaining X less than half: {len(x_less_after_inv)} individual tiles")
                print(f"Expanded X more than half: {len(x_more_expanded)} individual tiles")
                
                # Match pieces between apartments
                x_apt_matches, x_unmatched_less, x_unmatched_more = self.match_with_progressive_tolerance(
                    x_less_after_inv, x_more_expanded, tolerance_ranges, "X")
                print(f"Found {len(x_apt_matches)} X apartment matches")
                
                # Update counters
                for match in x_apt_matches:
                    if match.get('Same Apartment', False):
                        within_apartment_matches += 1
                    else:
                        cross_apartment_matches += 1
                    
                    # Update savings
                    total_savings += match.get('Small Piece Size', 0)
            else:
                # If either dataset is empty, all pieces are unmatched
                x_unmatched_less = x_less_after_inv
                if not x_more_than_half.empty:
                    x_unmatched_more = self.expand_dataframe(x_more_than_half)
                else:
                    x_unmatched_more = []
                reason = "No remaining less pieces" if len(x_less_after_inv) == 0 else "No more-than-half pieces"
                print(f"Skipping X apartment matching ({reason})")
            
            # Create summary DataFrames for X direction
            x_inv_matches_df = pd.DataFrame(x_inv_matches)
            x_apt_matches_df = pd.DataFrame(x_apt_matches)
            
            # Aggregate unmatched pieces
            x_unmatched_less_agg = self._aggregate_unmatched(x_unmatched_less)
            x_unmatched_more_agg = self._aggregate_unmatched(x_unmatched_more)
            
            # Store X direction results
            match_results['x_inv_matches'] = x_inv_matches_df
            match_results['x_apt_matches'] = x_apt_matches_df
            match_results['x_unmatched_less'] = x_unmatched_less_agg
            match_results['x_unmatched_more'] = x_unmatched_more_agg
            
            # =============== PROCESS Y DIRECTION ===============
            # Similar implementation as X direction but for Y pieces...
            # (Include the full Y direction processing here, similar to the X direction)
            
            # Process Y direction
            print("\n📋 Processing Cut Y Tiles:")
            y_less_than_half = cut_pieces_by_half.get('y_less_than_half', pd.DataFrame())
            y_more_than_half = cut_pieces_by_half.get('y_more_than_half', pd.DataFrame())
            
            if isinstance(y_less_than_half, list):
                y_less_than_half = pd.DataFrame(y_less_than_half) if y_less_than_half else pd.DataFrame()
            if isinstance(y_more_than_half, list):
                y_more_than_half = pd.DataFrame(y_more_than_half) if y_more_than_half else pd.DataFrame()
            
            print(f"Y less than half: {len(y_less_than_half)} rows")
            print(f"Y more than half: {len(y_more_than_half)} rows")
            
            # Track inventory matches and remaining pieces
            y_less_after_inv = []
            
            # First, try inventory matching if available
            if has_inventory and not y_less_than_half.empty and inventory_data_processed:
                print("\n🔄 Matching Cut Y Less than Half with Inventory...")
                
                # Get Y direction inventory data - check all possible keys
                y_inv_less = None
                if 'cut_y_inventory' in inventory_data_processed and not inventory_data_processed['cut_y_inventory'].empty:
                    y_inv_less = inventory_data_processed['cut_y_inventory']
                    print(f"Found cut_y_inventory with {len(y_inv_less)} rows")
                elif 'y_inv_less_than_half' in cut_pieces_by_half and not isinstance(cut_pieces_by_half['y_inv_less_than_half'], list) and not cut_pieces_by_half['y_inv_less_than_half'].empty:
                    y_inv_less = cut_pieces_by_half['y_inv_less_than_half']
                    print(f"Found y_inv_less_than_half with {len(y_inv_less)} rows")
                elif isinstance(inventory_data, dict) and 'cut_y_inventory' in inventory_data and not inventory_data['cut_y_inventory'].empty:
                    y_inv_less = inventory_data['cut_y_inventory']
                    print(f"Found cut_y_inventory directly in inventory_data with {len(y_inv_less)} rows")
                
                if y_inv_less is not None and not y_inv_less.empty:
                    # Expand DataFrames
                    y_less_expanded = self.expand_dataframe(y_less_than_half)
                    y_inv_expanded = self.expand_dataframe(y_inv_less)
                    
                    print(f"Expanded Y less than half: {len(y_less_expanded)} individual tiles")
                    print(f"Expanded Y inventory: {len(y_inv_expanded)} individual pieces")
                    
                    # Match with inventory
                    y_inv_matches, y_less_after_inv, _ = self.match_with_inventory(
                        y_less_expanded, y_inv_expanded, tolerance_ranges, "Y")
                    print(f"Found {len(y_inv_matches)} Y inventory matches")
                    
                    # Update counter
                    inventory_matches += len(y_inv_matches)
                    
                    # Update savings
                    for match in y_inv_matches:
                        total_savings += match.get('Cut Size (mm)', 0)
                else:
                    # No inventory data, use all less pieces for apartment matching
                    y_less_after_inv = self.expand_dataframe(y_less_than_half)
                    print("No suitable Y inventory data found, skipping inventory matching")
            else:
                # No inventory, use all less pieces for apartment matching
                if not y_less_than_half.empty:
                    y_less_after_inv = self.expand_dataframe(y_less_than_half)
                    reason = "No inventory provided" if not has_inventory else "Empty less_than_half" if y_less_than_half.empty else "No processed inventory data"
                    print(f"Skipping Y inventory matching ({reason})")
                else:
                    y_less_after_inv = []
                    print("No Y less-than-half pieces to match")
            
            # Now do apartment-to-apartment matching for remaining pieces
            y_unmatched_less = []
            y_unmatched_more = []
            
            if len(y_less_after_inv) > 0 and not y_more_than_half.empty:
                print("\n🔄 Matching remaining Cut Y pieces between apartments...")
                
                # Expand more than half pieces
                y_more_expanded = self.expand_dataframe(y_more_than_half)
                
                print(f"Remaining Y less than half: {len(y_less_after_inv)} individual tiles")
                print(f"Expanded Y more than half: {len(y_more_expanded)} individual tiles")
                
                # Match pieces between apartments
                y_apt_matches, y_unmatched_less, y_unmatched_more = self.match_with_progressive_tolerance(
                    y_less_after_inv, y_more_expanded, tolerance_ranges, "Y")
                print(f"Found {len(y_apt_matches)} Y apartment matches")
                
                # Update counters
                for match in y_apt_matches:
                    if match.get('Same Apartment', False):
                        within_apartment_matches += 1
                    else:
                        cross_apartment_matches += 1
                    
                    # Update savings
                    total_savings += match.get('Small Piece Size', 0)
            else:
                # If either dataset is empty, all pieces are unmatched
                y_unmatched_less = y_less_after_inv
                if not y_more_than_half.empty:
                    y_unmatched_more = self.expand_dataframe(y_more_than_half)
                else:
                    y_unmatched_more = []
                reason = "No remaining less pieces" if len(y_less_after_inv) == 0 else "No more-than-half pieces"
                print(f"Skipping Y apartment matching ({reason})")
            
            # Create summary DataFrames for Y direction
            y_inv_matches_df = pd.DataFrame(y_inv_matches)
            y_apt_matches_df = pd.DataFrame(y_apt_matches)
            
            # Aggregate unmatched pieces
            y_unmatched_less_agg = self._aggregate_unmatched(y_unmatched_less)
            y_unmatched_more_agg = self._aggregate_unmatched(y_unmatched_more)
            
            # Store Y direction results
            match_results['y_inv_matches'] = y_inv_matches_df
            match_results['y_apt_matches'] = y_apt_matches_df
            match_results['y_unmatched_less'] = y_unmatched_less_agg
            match_results['y_unmatched_more'] = y_unmatched_more_agg
            
        else:
            # -------------- NO PATTERN MODE (ALL CUTS) --------------
            # (Include the all cuts implementation here)
            
            # Process All direction (no pattern)
            print("\n📋 Processing All Cut Tiles:")
            all_less_than_half = cut_pieces_by_half.get('all_less_than_half', pd.DataFrame())
            all_more_than_half = cut_pieces_by_half.get('all_more_than_half', pd.DataFrame())
            
            if isinstance(all_less_than_half, list):
                all_less_than_half = pd.DataFrame(all_less_than_half) if all_less_than_half else pd.DataFrame()
            if isinstance(all_more_than_half, list):
                all_more_than_half = pd.DataFrame(all_more_than_half) if all_more_than_half else pd.DataFrame()
            
            print(f"All less than half: {len(all_less_than_half)} rows")
            print(f"All more than half: {len(all_more_than_half)} rows")
            
            # Track inventory matches and remaining pieces
            all_less_after_inv = []
            
            # Process with inventory if available
            # Process with inventory if available
            if has_inventory and not all_less_than_half.empty and inventory_data_processed:
                print("\n🔄 Matching All Cut Less than Half with Inventory...")
                
                # Get All direction inventory data - check all possible keys
                all_inv_less = None
                if 'all_cut_inventory' in inventory_data_processed and not inventory_data_processed['all_cut_inventory'].empty:
                    all_inv_less = inventory_data_processed['all_cut_inventory']
                    print(f"Found all_cut_inventory with {len(all_inv_less)} rows")
                elif 'all_inv_less_than_half' in cut_pieces_by_half and not isinstance(cut_pieces_by_half['all_inv_less_than_half'], list) and not cut_pieces_by_half['all_inv_less_than_half'].empty:
                    all_inv_less = cut_pieces_by_half['all_inv_less_than_half']
                    print(f"Found all_inv_less_than_half with {len(all_inv_less)} rows")
                elif isinstance(inventory_data, dict) and 'all_cut_inventory' in inventory_data and not inventory_data['all_cut_inventory'].empty:
                    all_inv_less = inventory_data['all_cut_inventory']
                    print(f"Found all_cut_inventory directly in inventory_data with {len(all_inv_less)} rows")
                
                if all_inv_less is not None and not all_inv_less.empty:
                    # Expand DataFrames
                    all_less_expanded = self.expand_dataframe(all_less_than_half)
                    all_inv_expanded = self.expand_dataframe(all_inv_less)
                    
                    print(f"Expanded All less than half: {len(all_less_expanded)} individual tiles")
                    print(f"Expanded All inventory: {len(all_inv_expanded)} individual pieces")
                    
                    # Match with inventory
                    all_inv_matches, all_less_after_inv, _ = self.match_with_inventory(
                        all_less_expanded, all_inv_expanded, tolerance_ranges, "All")
                    print(f"Found {len(all_inv_matches)} All inventory matches")
                    
                    # Update counter
                    inventory_matches += len(all_inv_matches)
                    
                    # Update savings
                    for match in all_inv_matches:
                        total_savings += match.get('Cut Size (mm)', 0)
                else:
                    # No inventory data, use all less pieces for apartment matching
                    all_less_after_inv = self.expand_dataframe(all_less_than_half)
                    print("No suitable All inventory data found, skipping inventory matching")
            else:
                # No inventory, use all less pieces for apartment matching
                if not all_less_than_half.empty:
                    all_less_after_inv = self.expand_dataframe(all_less_than_half)
                    reason = "No inventory provided" if not has_inventory else "Empty less_than_half" if all_less_than_half.empty else "No processed inventory data"
                    print(f"Skipping All inventory matching ({reason})")
                else:
                    all_less_after_inv = []
                    print("No All less-than-half pieces to match")
            
            # Now do apartment-to-apartment matching
            all_unmatched_less = []
            all_unmatched_more = []
            
            if len(all_less_after_inv) > 0 and not all_more_than_half.empty:
                print("\n🔄 Matching remaining All Cut pieces between apartments...")
                
                # Expand more than half pieces
                all_more_expanded = self.expand_dataframe(all_more_than_half)
                
                print(f"Remaining All less than half: {len(all_less_after_inv)} individual tiles")
                print(f"Expanded All more than half: {len(all_more_expanded)} individual tiles")
                
                # Match using progressive tolerance
                all_apt_matches, all_unmatched_less, all_unmatched_more = self.match_with_progressive_tolerance(
                    all_less_after_inv, all_more_expanded, tolerance_ranges, "All")
                
                # Create matched dataframe
                all_apt_matches_df = pd.DataFrame(all_apt_matches)
                
                # Update counters
                for match in all_apt_matches:
                    if match.get('Same Apartment', False):
                        within_apartment_matches += 1
                    else:
                        cross_apartment_matches += 1
                    
                    # Update savings
                    total_savings += match.get('Small Piece Size', 0)
            else:
                # If either dataset is empty, all pieces are unmatched
                all_unmatched_less = all_less_after_inv
                if not all_more_than_half.empty:
                    all_unmatched_more = self.expand_dataframe(all_more_than_half)
                else:
                    all_unmatched_more = []
                reason = "No remaining less pieces" if len(all_less_after_inv) == 0 else "No more-than-half pieces"
                print(f"Skipping All apartment matching ({reason})")
            
            # Create summary DataFrames for All direction
            all_inv_matches_df = pd.DataFrame(all_inv_matches)
            all_apt_matches_df = pd.DataFrame(all_apt_matches)
            
            # Aggregate unmatched pieces
            all_unmatched_less_agg = self._aggregate_unmatched(all_unmatched_less)
            all_unmatched_more_agg = self._aggregate_unmatched(all_unmatched_more)
            
            # Store All direction results
            match_results['all_inv_matches'] = all_inv_matches_df
            match_results['all_apt_matches'] = all_apt_matches_df
            match_results['all_unmatched_less'] = all_unmatched_less_agg
            match_results['all_unmatched_more'] = all_unmatched_more_agg
        
        # Store final statistics in match_results
        match_results['matched_count'] = within_apartment_matches + cross_apartment_matches + inventory_matches
        match_results['within_apartment_matches'] = within_apartment_matches
        match_results['cross_apartment_matches'] = cross_apartment_matches
        match_results['inventory_matches'] = inventory_matches
        match_results['total_savings'] = float(total_savings)
        
        # Create simplified matches for display in UI
        simplified_matches = []
        
        if has_pattern:
            # Convert X and Y matches to display-friendly format
            # X apartment matches
            for _, match in x_apt_matches_df.iterrows():
                simplified_matches.append({
                    'Match ID': match.get('Match ID', ''),
                    'From': f"{match.get('Small Piece Apt', '')}-{match.get('Small Piece Loc', '')}",
                    'To': f"{match.get('Large Piece Apt', '')}-{match.get('Large Piece Loc', '')}",
                    'Size (mm)': match.get('Small Piece Size', 0),
                    'Waste (mm)': match.get('Waste', 0),
                    'Match Type': 'Same Apartment' if match.get('Same Apartment', False) else 'Cross Apartment'
                })
            
            # X inventory matches
            for _, match in x_inv_matches_df.iterrows():
                simplified_matches.append({
                    'Match ID': match.get('Match ID', ''),
                    'From': f"{match.get('Apartment', '')}-{match.get('Location', '')}",
                    'To': f"Inventory-{match.get('Inventory Location', '')}",
                    'Size (mm)': match.get('Cut Size (mm)', 0),
                    'Waste (mm)': match.get('Waste', 0),
                    'Match Type': 'Inventory'
                })
            
            # Y apartment matches
            for _, match in y_apt_matches_df.iterrows():
                simplified_matches.append({
                    'Match ID': match.get('Match ID', ''),
                    'From': f"{match.get('Small Piece Apt', '')}-{match.get('Small Piece Loc', '')}",
                    'To': f"{match.get('Large Piece Apt', '')}-{match.get('Large Piece Loc', '')}",
                    'Size (mm)': match.get('Small Piece Size', 0),
                    'Waste (mm)': match.get('Waste', 0),
                    'Match Type': 'Same Apartment' if match.get('Same Apartment', False) else 'Cross Apartment'
                })
            
            # Y inventory matches
            for _, match in y_inv_matches_df.iterrows():
                simplified_matches.append({
                    'Match ID': match.get('Match ID', ''),
                    'From': f"{match.get('Apartment', '')}-{match.get('Location', '')}",
                    'To': f"Inventory-{match.get('Inventory Location', '')}",
                    'Size (mm)': match.get('Cut Size (mm)', 0),
                    'Waste (mm)': match.get('Waste', 0),
                    'Match Type': 'Inventory'
                })
        else:
            # All cut matches
            for _, match in all_apt_matches_df.iterrows():
                simplified_matches.append({
                    'Match ID': match.get('Match ID', ''),
                    'From': f"{match.get('Small Piece Apt', '')}-{match.get('Small Piece Loc', '')}",
                    'To': f"{match.get('Large Piece Apt', '')}-{match.get('Large Piece Loc', '')}",
                    'Size (mm)': match.get('Small Piece Size', 0),
                    'Waste (mm)': match.get('Waste', 0),
                    'Match Type': 'Same Apartment' if match.get('Same Apartment', False) else 'Cross Apartment'
                })
            
            # All cut inventory matches
            for _, match in all_inv_matches_df.iterrows():
                simplified_matches.append({
                    'Match ID': match.get('Match ID', ''),
                    'From': f"{match.get('Apartment', '')}-{match.get('Location', '')}",
                    'To': f"Inventory-{match.get('Inventory Location', '')}",
                    'Size (mm)': match.get('Cut Size (mm)', 0),
                    'Waste (mm)': match.get('Waste', 0),
                    'Match Type': 'Inventory'
                })
        
        match_results['simplified_matches'] = simplified_matches
        
        # Print summary info
        print("\n===== MATCH SUMMARY =====")
        print(f"Total Matches: {match_results['matched_count']}")
        print(f"  - Same Apartment: {match_results['within_apartment_matches']}")
        print(f"  - Cross Apartment: {match_results['cross_apartment_matches']}")
        print(f"  - Inventory: {match_results['inventory_matches']}")
        print(f"Total Material Saved: {match_results['total_savings']} mm²")
        
        return match_results
    
    def _aggregate_unmatched(self, unmatched_pieces):
        """
        Aggregate unmatched pieces by apartment, location, and size
        
        Parameters:
        - unmatched_pieces: List of unmatched pieces
        
        Returns:
        - DataFrame with aggregated unmatched pieces
        """
        if not unmatched_pieces:
            return pd.DataFrame()
            
        # Group by key attributes
        aggregated = {}
        for piece in unmatched_pieces:
            key = (
                piece.get('Apartment', ''),
                piece.get('Location', ''),
                piece.get('Cut Size (mm)', 0),
                piece.get('Remaining Size (mm)', 0)
            )
            if key not in aggregated:
                aggregated[key] = 0
            aggregated[key] += 1
        
        # Convert to DataFrame
        result = []
        for key, count in aggregated.items():
            apartment, location, cut_size, remaining_size = key
            result.append({
                'Apartment': apartment,
                'Location': location,
                'Cut Size (mm)': cut_size,
                'Remaining Size (mm)': remaining_size,
                'Count': count
            })
        
        return pd.DataFrame(result)
    
    def generate_match_visualization(self, match_results, room_df):
        """
        Generate a visualization of matched cut pieces
        
        Parameters:
        -----------
        match_results : dict
            Results from the matching process
        room_df : DataFrame
            DataFrame with room information
            
        Returns:
        --------
        str
            Base64 encoded image
        """
        import matplotlib.pyplot as plt
        import io
        import base64
        import numpy as np
        from matplotlib.patches import Patch
        
        # Create figure
        plt.figure(figsize=(16, 12))
        
        # Define colors for different match types
        SAME_APT_COLORS = [
            '#FFD700', '#FFA500', '#FF6347', '#FF1493', '#9932CC', 
            '#4169E1', '#00BFFF', '#00FA9A', '#ADFF2F', '#FFD700'
        ]  # Multiple colors for same apartment matches
        DIFF_APT_COLOR = '#A9A9A9'  # Gray for different apartment matches
        INV_COLOR = '#8FBC8F'       # Green for inventory matches
        
        # Plot room outlines
        for _, room in room_df.iterrows():
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
        
        # Track counts for legend
        classification_counts = {'full': 0, 'irregular': 0, 'cut_x': 0, 'cut_y': 0, 'all_cut': 0, 'total': 0}
        
        # Get simplified matches
        simplified_matches = match_results.get('simplified_matches', [])
        match_count = len(simplified_matches)
        
        # Add highlighted colored areas for matches
        for i, match in enumerate(simplified_matches):
            match_type = match.get('Match Type', '')
            match_id = match.get('Match ID', '')
            
            # Choose color based on match type
            if match_type == 'Inventory':
                color = INV_COLOR
            elif match_type == 'Same Apartment':
                # Extract number from match_id (X1, Y2, etc.)
                match_num = 1
                import re
                num_match = re.search(r'\d+', match_id)
                if num_match:
                    match_num = int(num_match.group())
                color = SAME_APT_COLORS[(match_num - 1) % len(SAME_APT_COLORS)]
            else:  # Cross Apartment
                color = DIFF_APT_COLOR
            
            # Draw a simple placeholder representation
            # In a real implementation, we would draw actual tiles based on their geometry
            # Here we'll just add some visual indicators on the plot
            
            # Select random position within plot bounds
            # (This is just a placeholder - in a real implementation, use actual coordinates)
            x_range = plt.xlim()
            y_range = plt.ylim()
            center_x = np.random.uniform(x_range[0] + (x_range[1]-x_range[0])*0.1, 
                                        x_range[1] - (x_range[1]-x_range[0])*0.1)
            center_y = np.random.uniform(y_range[0] + (y_range[1]-y_range[0])*0.1, 
                                        y_range[1] - (y_range[1]-y_range[0])*0.1)
            
            # Draw a colored circle to represent the match
            plt.scatter(center_x, center_y, s=100, color=color, alpha=0.7, edgecolor='black')
            
            # Add text label with match ID
            plt.text(center_x, center_y + (y_range[1]-y_range[0])*0.02, match_id, 
                    fontsize=8, ha='center', va='center', color='black')
        
        # Add legend with counts
        legend_elements = [
            Patch(facecolor=SAME_APT_COLORS[0], alpha=0.7, edgecolor='black', 
                label=f"Same Apartment Match ({match_results.get('within_apartment_matches', 0)})"),
            Patch(facecolor=DIFF_APT_COLOR, alpha=0.7, edgecolor='black', 
                label=f"Cross Apartment Match ({match_results.get('cross_apartment_matches', 0)})"),
            Patch(facecolor=INV_COLOR, alpha=0.7, edgecolor='black', 
                label=f"Inventory Match ({match_results.get('inventory_matches', 0)})")
        ]
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add stats in a text box
        within_apt = match_results.get('within_apartment_matches', 0)
        cross_apt = match_results.get('cross_apartment_matches', 0)
        inv_matches = match_results.get('inventory_matches', 0)
        total_matches = within_apt + cross_apt + inv_matches
        savings = match_results.get('total_savings', 0)
        
        stats_text = (
            f"Total Matches: {total_matches}\n"
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
    
    def create_inventory_template(self, has_pattern=True):
        """
        Create inventory template Excel file
        
        Parameters:
        - has_pattern: Whether to use pattern mode
        
        Returns:
        - BytesIO object with Excel file
        """
        import io
        import pandas as pd
        
        # Create an in-memory output file
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            if has_pattern:
                # Create Cut X template
                cut_x_template = pd.DataFrame(columns=[
                    'Remaining Size (mm)', 'Location', 'Count'
                ])
                cut_x_template.to_excel(writer, sheet_name='Cut X', index=False)
                
                # Create Cut Y template
                cut_y_template = pd.DataFrame(columns=[
                    'Remaining Size (mm)', 'Location', 'Count'
                ])
                cut_y_template.to_excel(writer, sheet_name='Cut Y', index=False)
            else:
                # Create All Cut template
                all_cut_template = pd.DataFrame(columns=[
                    'Remaining Size (mm)', 'Location', 'Count'
                ])
                all_cut_template.to_excel(writer, sheet_name='All Cut', index=False)
            
            # Add sample data
            sample_data = pd.DataFrame([
                {'Remaining Size (mm)': 150, 'Location': 'INV', 'Count': 2},
                {'Remaining Size (mm)': 220, 'Location': 'INV', 'Count': 3},
                {'Remaining Size (mm)': 350, 'Location': 'INV', 'Count': 1},
                {'Remaining Size (mm)': 480, 'Location': 'INV', 'Count': 4}
            ])
            sample_data.to_excel(writer, sheet_name='Sample Data', index=False)
            
            # Add instructions
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
    
    def load_and_process_inventory(self, file):
        """
        Load and process inventory file
        
        Parameters:
        - file: Uploaded file (FileStorage)
        
        Returns:
        - Dict with processed inventory data
        """
        import pandas as pd
        
        try:
            # Check file type
            if file.filename.endswith('.xlsx'):
                # Read Excel file
                xl = pd.ExcelFile(file)
                
                # Check for pattern-based inventory
                has_pattern = 'Cut X' in xl.sheet_names and 'Cut Y' in xl.sheet_names
                all_cut = 'All Cut' in xl.sheet_names
                
                if not has_pattern and not all_cut:
                    return {'error': 'Invalid inventory file format. Expected sheets "Cut X" and "Cut Y" or "All Cut".'}
                
                inventory_data = {'has_pattern': has_pattern}
                
                if has_pattern:
                    # Read Cut X inventory
                    if 'Cut X' in xl.sheet_names:
                        cut_x_df = pd.read_excel(file, sheet_name='Cut X')
                        if not all(col in cut_x_df.columns for col in ['Remaining Size (mm)', 'Location', 'Count']):
                            return {'error': 'Cut X sheet missing required columns: "Remaining Size (mm)", "Location", "Count"'}
                        inventory_data['cut_x_inventory'] = cut_x_df
                    
                    # Read Cut Y inventory
                    if 'Cut Y' in xl.sheet_names:
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
    
    def generate_optimization_report(self, optimization_results, inventory_data=None):
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
        matches_df = pd.DataFrame(optimization_results.get('simplified_matches', []))
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Create summary sheet
            summary_data = {
                'Metric': [
                    'Total Matches',
                    'Within-Apartment Matches',
                    'Cross-Apartment Matches',
                    'Inventory Matches',
                    'Total Material Saved (mm²)',
                    'Total Material Saved (m²)',
                    'Optimization Date'
                ],
                'Value': [
                    optimization_results.get('matched_count', 0),
                    optimization_results.get('within_apartment_matches', 0),
                    optimization_results.get('cross_apartment_matches', 0),
                    optimization_results.get('inventory_matches', 0),
                    optimization_results.get('total_savings', 0),
                    optimization_results.get('total_savings', 0) / 1000000,
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
                    'criteria': '=$E2="Cross Apartment"',
                    'format': writer.book.add_format({'bg_color': '#ffffcc'})
                })
            
            # Create inventory sheet if inventory data is available
            if inventory_data:
                has_pattern = inventory_data.get('has_pattern', True)
                
                if has_pattern:
                    # For pattern-based inventory
                    x_inventory = inventory_data.get('cut_x_inventory', None)
                    if isinstance(x_inventory, list):
                        x_inventory = pd.DataFrame(x_inventory) if x_inventory else pd.DataFrame()
                    
                    y_inventory = inventory_data.get('cut_y_inventory', None)
                    if isinstance(y_inventory, list):
                        y_inventory = pd.DataFrame(y_inventory) if y_inventory else pd.DataFrame()
                    
                    if x_inventory is not None and not isinstance(x_inventory, pd.DataFrame):
                        # Try converting dictionary format
                        try:
                            x_inventory = pd.DataFrame(x_inventory)
                        except:
                            x_inventory = pd.DataFrame()
                    
                    if y_inventory is not None and not isinstance(y_inventory, pd.DataFrame):
                        # Try converting dictionary format
                        try:
                            y_inventory = pd.DataFrame(y_inventory)
                        except:
                            y_inventory = pd.DataFrame()
                    
                    if x_inventory is not None and not x_inventory.empty:
                        x_inventory.to_excel(writer, sheet_name='X Inventory', index=False)
                        x_sheet = writer.sheets['X Inventory']
                        x_sheet.set_column('A:C', 15)
                    
                    if y_inventory is not None and not y_inventory.empty:
                        y_inventory.to_excel(writer, sheet_name='Y Inventory', index=False)
                        y_sheet = writer.sheets['Y Inventory']
                        y_sheet.set_column('A:C', 15)
                else:
                    # For non-pattern inventory
                    all_inventory = inventory_data.get('all_cut_inventory', None)
                    if isinstance(all_inventory, list):
                        all_inventory = pd.DataFrame(all_inventory) if all_inventory else pd.DataFrame()
                    
                    if all_inventory is not None and not isinstance(all_inventory, pd.DataFrame):
                        # Try converting dictionary format
                        try:
                            all_inventory = pd.DataFrame(all_inventory)
                        except:
                            all_inventory = pd.DataFrame()
                    
                    if all_inventory is not None and not all_inventory.empty:
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
                    '3. Inventory - Inventory data used for optimization (if available)',
                    '',
                    'Match Types:',
                    '- Within-Apartment: Matches between pieces in the same apartment',
                    '- Cross-Apartment: Matches between pieces in different apartments (highlighted in yellow)',
                    '- Inventory: Matches between apartment pieces and inventory',
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
    
    def manual_grouping(self, cut_pieces_by_half, grouping_instructions):
        """
        Apply manual grouping to cut pieces based on user instructions
        
        Parameters:
        - cut_pieces_by_half: Dict with cut pieces split by half threshold
        - grouping_instructions: Dict with manual grouping instructions
        
        Returns:
        - Dict with manually grouped cut pieces
        """
        # Create a copy of the input data
        grouped_pieces = cut_pieces_by_half.copy()
        
        # Get pattern mode
        has_pattern = cut_pieces_by_half.get('has_pattern', True)
        
        # Process grouping instructions
        for group in grouping_instructions:
            group_id = group.get('group_id')
            apartments = group.get('apartments', [])
            directions = group.get('directions', [])
            
            if not group_id or not apartments:
                continue
            
            # Apply grouping based on pattern mode
            if has_pattern:
                # Apply to X direction if specified
                if 'X' in directions or not directions:
                    self._apply_group_to_dataframe(grouped_pieces, 'x_less_than_half', group_id, apartments)
                    self._apply_group_to_dataframe(grouped_pieces, 'x_more_than_half', group_id, apartments)
                
                # Apply to Y direction if specified
                if 'Y' in directions or not directions:
                    self._apply_group_to_dataframe(grouped_pieces, 'y_less_than_half', group_id, apartments)
                    self._apply_group_to_dataframe(grouped_pieces, 'y_more_than_half', group_id, apartments)
            else:
                # Apply to all cuts
                self._apply_group_to_dataframe(grouped_pieces, 'all_less_than_half', group_id, apartments)
                self._apply_group_to_dataframe(grouped_pieces, 'all_more_than_half', group_id, apartments)
        
        return grouped_pieces

    def _apply_group_to_dataframe(self, data_dict, key, group_id, apartments):
        """Helper method to apply group ID to pieces from specific apartments"""
        if key not in data_dict or data_dict[key].empty:
            return
        
        # Get the DataFrame
        df = data_dict[key]
        
        # Add group_id column if it doesn't exist
        if 'Group ID' not in df.columns:
            df['Group ID'] = None
        
        # Update group ID for selected apartments
        mask = df['Apartment'].isin(apartments)
        df.loc[mask, 'Group ID'] = group_id
        
        # Update the DataFrame in the dictionary
        data_dict[key] = df

    def export_matching_results_to_excel(self, match_results):
        """
        Export matching results to Excel
        
        Parameters:
        - match_results: Dict with matching results
        
        Returns:
        - BytesIO object with Excel file
        """
        import io
        import pandas as pd
        from datetime import datetime
        
        # Create an in-memory output file
        output = io.BytesIO()
        
        # Get pattern mode
        has_pattern = match_results.get('has_pattern', True)
        
        # Convert matches to DataFrame
        simplified_matches = match_results.get('simplified_matches', [])
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Create summary sheet
            summary_data = {
                'Metric': [
                    'Total Matches',
                    'Within-Apartment Matches',
                    'Cross-Apartment Matches',
                    'Inventory Matches',
                    'Total Material Saved (mm²)',
                    'Total Material Saved (m²)',
                    'Optimization Date'
                ],
                'Value': [
                    match_results.get('matched_count', 0),
                    match_results.get('within_apartment_matches', 0),
                    match_results.get('cross_apartment_matches', 0),
                    match_results.get('inventory_matches', 0),
                    match_results.get('total_savings', 0),
                    match_results.get('total_savings', 0) / 1000000,
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
            matches_df = pd.DataFrame(simplified_matches)
            if not matches_df.empty:
                matches_df.to_excel(writer, sheet_name='Matches', index=False)
                
                # Format the matches sheet
                matches_sheet = writer.sheets['Matches']
                matches_sheet.set_column('A:E', 15)
                
                # Add conditional formatting for cross-apartment matches
                matches_sheet.conditional_format('A2:E1000', {
                    'type': 'formula',
                    'criteria': '=$E2="Cross Apartment"',
                    'format': writer.book.add_format({'bg_color': '#ffffcc'})
                })
            
            # Create inventory sheet if inventory data is available
            if 'inventory_data' in match_results:
                has_pattern = match_results.get('has_pattern', True)
                inventory_data = match_results.get('inventory_data', {})
                
                if has_pattern:
                    # For pattern-based inventory
                    x_inventory = inventory_data.get('cut_x_inventory', pd.DataFrame())
                    if isinstance(x_inventory, list):
                        x_inventory = pd.DataFrame(x_inventory) if x_inventory else pd.DataFrame()
                    
                    y_inventory = inventory_data.get('cut_y_inventory', pd.DataFrame())
                    if isinstance(y_inventory, list):
                        y_inventory = pd.DataFrame(y_inventory) if y_inventory else pd.DataFrame()
                    
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
                    if isinstance(all_inventory, list):
                        all_inventory = pd.DataFrame(all_inventory) if all_inventory else pd.DataFrame()
                    
                    if not all_inventory.empty:
                        all_inventory.to_excel(writer, sheet_name='All Inventory', index=False)
                        all_sheet = writer.sheets['All Inventory']
                        all_sheet.set_column('A:C', 15)
            
            # Create unmatched pieces sheets
            if has_pattern:
                # X unmatched pieces
                x_unmatched_less = match_results.get('x_unmatched_less', pd.DataFrame())
                x_unmatched_more = match_results.get('x_unmatched_more', pd.DataFrame())
                
                if isinstance(x_unmatched_less, list):
                    x_unmatched_less = pd.DataFrame(x_unmatched_less) if x_unmatched_less else pd.DataFrame()
                if isinstance(x_unmatched_more, list):
                    x_unmatched_more = pd.DataFrame(x_unmatched_more) if x_unmatched_more else pd.DataFrame()
                
                if not x_unmatched_less.empty or not x_unmatched_more.empty:
                    # Add piece type column
                    if not x_unmatched_less.empty:
                        x_unmatched_less['Piece Type'] = 'Less than Half'
                    if not x_unmatched_more.empty:
                        x_unmatched_more['Piece Type'] = 'More than Half'
                    
                    # Combine and export
                    x_unmatched = pd.concat([x_unmatched_less, x_unmatched_more])
                    if not x_unmatched.empty:
                        x_unmatched.to_excel(writer, sheet_name='X Unmatched', index=False)
                        x_unmatched_sheet = writer.sheets['X Unmatched']
                        x_unmatched_sheet.set_column('A:E', 15)
                
                # Y unmatched pieces
                y_unmatched_less = match_results.get('y_unmatched_less', pd.DataFrame())
                y_unmatched_more = match_results.get('y_unmatched_more', pd.DataFrame())
                
                if isinstance(y_unmatched_less, list):
                    y_unmatched_less = pd.DataFrame(y_unmatched_less) if y_unmatched_less else pd.DataFrame()
                if isinstance(y_unmatched_more, list):
                    y_unmatched_more = pd.DataFrame(y_unmatched_more) if y_unmatched_more else pd.DataFrame()
                
                if not y_unmatched_less.empty or not y_unmatched_more.empty:
                    # Add piece type column
                    if not y_unmatched_less.empty:
                        y_unmatched_less['Piece Type'] = 'Less than Half'
                    if not y_unmatched_more.empty:
                        y_unmatched_more['Piece Type'] = 'More than Half'
                    
                    # Combine and export
                    y_unmatched = pd.concat([y_unmatched_less, y_unmatched_more])
                    if not y_unmatched.empty:
                        y_unmatched.to_excel(writer, sheet_name='Y Unmatched', index=False)
                        y_unmatched_sheet = writer.sheets['Y Unmatched']
                        y_unmatched_sheet.set_column('A:E', 15)
            else:
                # All unmatched pieces
                all_unmatched_less = match_results.get('all_unmatched_less', pd.DataFrame())
                all_unmatched_more = match_results.get('all_unmatched_more', pd.DataFrame())
                
                if isinstance(all_unmatched_less, list):
                    all_unmatched_less = pd.DataFrame(all_unmatched_less) if all_unmatched_less else pd.DataFrame()
                if isinstance(all_unmatched_more, list):
                    all_unmatched_more = pd.DataFrame(all_unmatched_more) if all_unmatched_more else pd.DataFrame()
                
                if not all_unmatched_less.empty or not all_unmatched_more.empty:
                    # Add piece type column
                    if not all_unmatched_less.empty:
                        all_unmatched_less['Piece Type'] = 'Less than Half'
                    if not all_unmatched_more.empty:
                        all_unmatched_more['Piece Type'] = 'More than Half'
                    
                    # Combine and export
                    all_unmatched = pd.concat([all_unmatched_less, all_unmatched_more])
                    if not all_unmatched.empty:
                        all_unmatched.to_excel(writer, sheet_name='All Unmatched', index=False)
                        all_unmatched_sheet = writer.sheets['All Unmatched']
                        all_unmatched_sheet.set_column('A:E', 15)
            
            # Create instructions sheet
            instructions_data = {
                'Instructions': [
                    'This report contains the results of tile cut piece optimization.',
                    '',
                    'Sheets in this workbook:',
                    '1. Summary - Overall optimization statistics',
                    '2. Matches - Detailed list of all matched cut pieces',
                    '3. X/Y/All Inventory - Inventory data used for optimization (if available)',
                    '4. X/Y/All Unmatched - Unmatched pieces remaining after optimization',
                    '',
                    'Match Types:',
                    '- Within-Apartment: Matches between pieces in the same apartment',
                    '- Cross-Apartment: Matches between pieces in different apartments (highlighted in yellow)',
                    '- Inventory: Matches between apartment pieces and inventory',
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
        
    
    
    def download_full_report(self, match_results, inventory_data=None):
        """
        Generate a complete report package including match results and visualizations
        
        Parameters:
        - match_results: Dict with match results
        - inventory_data: Dict with inventory data (optional)
        
        Returns:
        - BytesIO object with ZIP file
        """
        import io
        import zipfile
        from datetime import datetime
        
        # Create an in-memory output file for the zip
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            # Add matching results Excel file
            matching_excel = self.export_matching_results_to_excel(match_results)
            zip_file.writestr('Matching_Results.xlsx', matching_excel.getvalue())
            
            # Add optimization report
            optimization_report = self.generate_optimization_report(match_results, inventory_data)
            zip_file.writestr('Optimization_Report.xlsx', optimization_report.getvalue())
            
            # Add README text file
            readme_content = (
                "Tile Matching Optimization Report\n"
                "================================\n\n"
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"Total Matches: {match_results.get('matched_count', 0)}\n"
                f"Within-Apartment Matches: {match_results.get('within_apartment_matches', 0)}\n"
                f"Cross-Apartment Matches: {match_results.get('cross_apartment_matches', 0)}\n"
                f"Inventory Matches: {match_results.get('inventory_matches', 0)}\n"
                f"Total Material Saved: {match_results.get('total_savings', 0)} mm² "
                f"({match_results.get('total_savings', 0) / 1000000:.3f} m²)\n\n"
                "This package contains:\n"
                "- Matching_Results.xlsx: Detailed matching results\n"
                "- Optimization_Report.xlsx: Summary report with statistics\n"
            )
            zip_file.writestr('README.txt', readme_content)
            
            # If there's an optimization plot, include it
            if 'optimization_plot' in match_results and match_results['optimization_plot']:
                import base64
                try:
                    # Convert base64 string to image
                    plot_data = base64.b64decode(match_results['optimization_plot'])
                    zip_file.writestr('Optimization_Visualization.png', plot_data)
                    
                    # Update readme to mention the image
                    readme_content += "- Optimization_Visualization.png: Visualization of matches\n"
                    zip_file.writestr('README.txt', readme_content)
                except:
                    # If conversion fails, skip the image
                    pass
        
        # Reset the buffer position
        zip_buffer.seek(0)
        
        return zip_buffer