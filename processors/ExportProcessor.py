import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os
import zipfile
from datetime import datetime

class ExportProcessor:
    """Processor for exporting tile matching results in detailed files"""
    
    def __init__(self):
        """Initialize the export processor"""
        self.export_folder = None
        self.color_map = {
            'SAME_APT_COLORS': [
                '#FFD700', '#FFA500', '#FF6347', '#FF1493', '#9932CC', 
                '#4169E1', '#00BFFF', '#00FA9A', '#ADFF2F', '#FFD700'
            ],  # Multiple colors for same apartment matches
            'DIFF_APT_COLOR': '#A9A9A9',  # Gray for different apartment matches
            'INV_COLOR': '#8FBC8F'       # Green for inventory matches
        }

    def initialize_export_folder(self):
        """Create a folder for exports with timestamp"""
        # Create a timestamp for the folder name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.export_folder = f"exports/Tile_Matching_Reports_{timestamp}"
        
        # Make sure the parent directory exists
        os.makedirs(os.path.dirname(self.export_folder), exist_ok=True)
        
        # Create the export folder
        os.makedirs(self.export_folder, exist_ok=True)
        
        return self.export_folder
    
    def export_apartment_data(self, raw_df, apartment_id, has_pattern=True):
        """
        Export apartment-specific data to Excel
        
        Parameters:
        - raw_df: DataFrame with raw apartment data
        - apartment_id: ID of the apartment to export
        - has_pattern: Whether pattern mode is used (X/Y cuts)
        
        Returns:
        - Path to the created Excel file
        """
        print(f"Processing apartment {apartment_id}...")
        
        # Check if export folder exists
        if not self.export_folder:
            self.initialize_export_folder()
            
        # Filter data for this apartment only
        apt_data = raw_df[raw_df['Apartment'] == apartment_id].copy()
        
        # Skip if no data for this apartment
        if apt_data.empty:
            print(f"  - No data found for apartment {apartment_id}")
            return None
        
        # Create a filename for this apartment
        apt_file = os.path.join(self.export_folder, f"{apartment_id}_Tile_Matching.xlsx")
        
        # Format the data
        apt_data = apt_data.rename(columns={
            'Cut Size': 'TILE SIZE(mm)',
            'Location': 'LOCATION',
            'Count': 'COUNT',
            'Remaining Size': 'REMAINING PIECE(mm)',
            'Group ID': 'MATCH(Group ID)',
            'Remarks': 'COMMENTS'
        })
        
        # Define needed columns
        keep_columns = ['TILE SIZE(mm)', 'LOCATION', 'COUNT', 
                        'REMAINING PIECE(mm)', 'MATCH(Group ID)', 'COMMENTS', 
                        'Piece Type', 'Match Type', 'Color']
        
        # Keep only relevant columns that exist
        keep_existing = [col for col in keep_columns if col in apt_data.columns]
        apt_data = apt_data[keep_existing]
        
        # Filter data for X and Y directions based on Group ID
        if has_pattern:
            x_data = apt_data[apt_data['MATCH(Group ID)'].astype(str).str.startswith(('X', 'OX', 'IX'))].copy() if 'MATCH(Group ID)' in apt_data.columns else pd.DataFrame()
            y_data = apt_data[apt_data['MATCH(Group ID)'].astype(str).str.startswith(('Y', 'OY', 'IY'))].copy() if 'MATCH(Group ID)' in apt_data.columns else pd.DataFrame()
            unmatched = apt_data[apt_data['Match Type'] == 'Unmatched'].copy() if 'Match Type' in apt_data.columns else pd.DataFrame()
            
            # Add unmatched to both X and Y
            if not unmatched.empty:
                x_data = pd.concat([x_data, unmatched]) if not x_data.empty else unmatched.copy()
                y_data = pd.concat([y_data, unmatched]) if not y_data.empty else unmatched.copy()
            
            # Sort by Tile Size
            if not x_data.empty:
                x_data = x_data.sort_values(by=['TILE SIZE(mm)'])
            
            if not y_data.empty:
                y_data = y_data.sort_values(by=['TILE SIZE(mm)'])
        else:
            # For non-pattern mode, use all data
            x_data = apt_data.copy()
            y_data = pd.DataFrame()  # Empty for non-pattern mode
            
            # Sort by Tile Size
            if not x_data.empty:
                x_data = x_data.sort_values(by=['TILE SIZE(mm)'])
        
        # Write to Excel with minimal formatting
        with pd.ExcelWriter(apt_file, engine='xlsxwriter') as writer:
            # Create X sheet (or All sheet for non-pattern)
            if not x_data.empty:
                sheet_name = 'X Direction' if has_pattern else 'All Cuts'
                x_data.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Basic column widths
                worksheet = writer.sheets[sheet_name]
                worksheet.set_column('A:A', 15)  # TILE SIZE
                worksheet.set_column('B:B', 10)  # LOCATION
                worksheet.set_column('C:C', 8)   # COUNT
                worksheet.set_column('D:D', 18)  # REMAINING PIECE
                worksheet.set_column('E:E', 15)  # MATCH
                worksheet.set_column('F:F', 50)  # COMMENTS
                
                print(f"  - Created {sheet_name} sheet with {len(x_data)} rows")
            
            # Create Y sheet if in pattern mode
            if has_pattern and not y_data.empty:
                y_data.to_excel(writer, sheet_name='Y Direction', index=False)
                
                # Basic column widths
                worksheet = writer.sheets['Y Direction']
                worksheet.set_column('A:A', 15)  # TILE SIZE
                worksheet.set_column('B:B', 10)  # LOCATION
                worksheet.set_column('C:C', 8)   # COUNT
                worksheet.set_column('D:D', 18)  # REMAINING PIECE
                worksheet.set_column('E:E', 15)  # MATCH
                worksheet.set_column('F:F', 50)  # COMMENTS
                
                print(f"  - Created Y Direction sheet with {len(y_data)} rows")
        
        print(f"  ✅ Excel file created: {apt_file}")
        return apt_file
    
    def export_inventory_data(self, inv_df, apt_unmatched_df=None, has_pattern=True):
        """
        Export inventory data to Excel
        
        Parameters:
        - inv_df: DataFrame with inventory data
        - apt_unmatched_df: DataFrame with unmatched apartment data
        - has_pattern: Whether pattern mode is used (X/Y cuts)
        
        Returns:
        - Path to the created Excel file
        """
        print("Processing inventory data...")

        # Check if export folder exists
        if not self.export_folder:
            self.initialize_export_folder()
        
        # Create a filename for inventory
        inv_file = os.path.join(self.export_folder, "Inventory_Tile_Matching.xlsx")
        
        # Format column names for inventory
        inv_data = inv_df.copy() if not inv_df.empty else pd.DataFrame()
        if not inv_data.empty:
            inv_data = inv_data.rename(columns={
                'Location': 'LOCATION',
                'Size (mm)': 'SIZE(mm)',
                'Status': 'STATUS',
                'Matched With': 'COMMENTS',
                'Group ID': 'MATCH(Group ID)'
            })
        
        # Prepare combined remaining inventory data
        # 1. Get unmatched inventory pieces
        inv_unmatched = pd.DataFrame()
        if not inv_data.empty and 'STATUS' in inv_data.columns:
            inv_unmatched = inv_data[inv_data['STATUS'] == 'Unmatched'].copy()
            if not inv_unmatched.empty:
                inv_unmatched['SOURCE'] = 'Inventory'
                
                # Standardize column names for merging
                inv_unmatched = inv_unmatched.rename(columns={
                    'SIZE(mm)': 'PIECE SIZE(mm)',
                    'Piece Type': 'PIECE TYPE'
                })
                
                # Keep only necessary columns
                inv_cols = ['LOCATION', 'PIECE SIZE(mm)', 'PIECE TYPE', 'SOURCE', 'COUNT'] 
                inv_cols = [col for col in inv_cols if col in inv_unmatched.columns]
                if 'COUNT' not in inv_cols:
                    inv_unmatched['COUNT'] = 1
                    inv_cols.append('COUNT')
                inv_unmatched = inv_unmatched[inv_cols]
        
        # 2. Get unmatched apartment pieces
        apt_unmatched = pd.DataFrame()
        if apt_unmatched_df is not None and not apt_unmatched_df.empty and 'Match Type' in apt_unmatched_df.columns:
            apt_unmatched = apt_unmatched_df[apt_unmatched_df['Match Type'] == 'Unmatched'].copy()
            
            if not apt_unmatched.empty:
                # Format column names
                apt_unmatched = apt_unmatched.rename(columns={
                    'Cut Size': 'TILE SIZE(mm)',
                    'Remaining Size': 'REMAINING PIECE(mm)',
                    'Location': 'LOCATION',
                    'Count': 'COUNT',
                    'Piece Type': 'PIECE TYPE'
                })
                
                # Create two entries for each row: one for the cut size and one for the remaining piece
                apt_cut = apt_unmatched.copy()
                apt_remain = apt_unmatched.copy()
                
                # For cut pieces, use the Cut Size and mark as "Cut Piece"
                apt_cut['PIECE SIZE(mm)'] = apt_cut['TILE SIZE(mm)']
                apt_cut['SOURCE'] = 'Apartment (Cut)'
                apt_cut['PIECE TYPE'] = apt_cut['PIECE TYPE'] + ' (Cut)'
                
                # For remaining pieces, use the Remaining Size and mark as "Remaining Piece"
                apt_remain['PIECE SIZE(mm)'] = apt_remain['REMAINING PIECE(mm)']
                apt_remain['SOURCE'] = 'Apartment (Remain)'
                apt_remain['PIECE TYPE'] = apt_remain['PIECE TYPE'] + ' (Remain)'
                
                # Keep only necessary columns
                apt_cols = ['Apartment', 'LOCATION', 'PIECE SIZE(mm)', 'PIECE TYPE', 'SOURCE', 'COUNT']
                apt_cols = [col for col in apt_cols if col in apt_cut.columns]
                apt_cut = apt_cut[apt_cols]
                apt_remain = apt_remain[apt_cols]
                
                # Combine cut and remaining pieces
                apt_unmatched = pd.concat([apt_cut, apt_remain])
        
        # 3. Combine all unmatched pieces
        all_unmatched = pd.concat([inv_unmatched, apt_unmatched]) if not inv_unmatched.empty or not apt_unmatched.empty else pd.DataFrame()
        
        # Write to Excel with minimal formatting
        with pd.ExcelWriter(inv_file, engine='xlsxwriter') as writer:
            # Add inventory sheets if we have inventory data
            if not inv_data.empty:
                if has_pattern and 'Direction' in inv_data.columns:
                    # Split by direction
                    x_inv = inv_data[inv_data['Direction'] == 'X'].copy() if 'Direction' in inv_data.columns else pd.DataFrame()
                    y_inv = inv_data[inv_data['Direction'] == 'Y'].copy() if 'Direction' in inv_data.columns else pd.DataFrame()
                    
                    # Add X inventory sheet
                    if not x_inv.empty:
                        x_inv.to_excel(writer, sheet_name='X Inventory', index=False)
                        worksheet = writer.sheets['X Inventory']
                        worksheet.set_column('A:A', 15)  # LOCATION
                        worksheet.set_column('B:B', 15)  # SIZE
                        worksheet.set_column('C:C', 10)  # STATUS
                        worksheet.set_column('D:D', 15)  # MATCH
                        worksheet.set_column('E:E', 50)  # COMMENTS
                        print(f"  - Created X Inventory sheet with {len(x_inv)} rows")
                    
                    # Add Y inventory sheet
                    if not y_inv.empty:
                        y_inv.to_excel(writer, sheet_name='Y Inventory', index=False)
                        worksheet = writer.sheets['Y Inventory']
                        worksheet.set_column('A:A', 15)  # LOCATION
                        worksheet.set_column('B:B', 15)  # SIZE
                        worksheet.set_column('C:C', 10)  # STATUS
                        worksheet.set_column('D:D', 15)  # MATCH
                        worksheet.set_column('E:E', 50)  # COMMENTS
                        print(f"  - Created Y Inventory sheet with {len(y_inv)} rows")
                
                # Add Combined Inventory sheet
                inv_data.to_excel(writer, sheet_name='Combined Inventory', index=False)
                worksheet = writer.sheets['Combined Inventory']
                worksheet.set_column('A:A', 15)  # First column
                worksheet.set_column('B:B', 15)  # Second column
                worksheet.set_column('C:C', 10)  # Third column
                worksheet.set_column('D:D', 15)  # Fourth column
                worksheet.set_column('E:E', 50)  # Fifth column
                print(f"  - Created Combined Inventory sheet with {len(inv_data)} rows")
            
            # Add Remaining Pieces sheet (unmatched from both apartment and inventory)
            if not all_unmatched.empty:
                # Sort by source, location, size
                all_unmatched = all_unmatched.sort_values(by=['SOURCE', 'LOCATION', 'PIECE SIZE(mm)'])
                
                # Write to sheet
                all_unmatched.to_excel(writer, sheet_name='Remaining Pieces', index=False)
                worksheet = writer.sheets['Remaining Pieces']
                
                # Set column widths based on actual columns
                for i, col in enumerate(all_unmatched.columns):
                    if col in ['Apartment', 'LOCATION', 'PIECE TYPE', 'SOURCE']:
                        worksheet.set_column(i, i, 15)  # Normal text columns
                    elif col in ['PIECE SIZE(mm)']:
                        worksheet.set_column(i, i, 18)  # Numeric columns
                    elif col in ['COUNT']:
                        worksheet.set_column(i, i, 8)   # Count column
                    else:
                        worksheet.set_column(i, i, 12)  # Default
                
                print(f"  - Created Remaining Pieces sheet with {len(all_unmatched)} rows")
            else:
                # Create an empty remaining pieces sheet
                empty_df = pd.DataFrame({
                    'LOCATION': [], 
                    'PIECE SIZE(mm)': [], 
                    'PIECE TYPE': [], 
                    'SOURCE': [], 
                    'COUNT': []
                })
                empty_df.to_excel(writer, sheet_name='Remaining Pieces', index=False)
                worksheet = writer.sheets['Remaining Pieces']
                worksheet.set_column('A:A', 15)  # LOCATION
                worksheet.set_column('B:B', 18)  # PIECE SIZE
                worksheet.set_column('C:C', 15)  # PIECE TYPE
                worksheet.set_column('D:D', 15)  # SOURCE
                worksheet.set_column('E:E', 8)   # COUNT
                print(f"  - Created empty Remaining Pieces sheet")
        
        print(f"  ✅ Excel file created: {inv_file}")
        return inv_file
    
    def create_summary_report(self, apt_files, inv_file, match_results, has_pattern=True, has_inventory=False):
        """
        Create a summary report of matching results
        
        Parameters:
        - apt_files: List of apartment Excel files
        - inv_file: Path to inventory Excel file
        - match_results: Dictionary with match results
        - has_pattern: Whether pattern mode is used (X/Y cuts)
        - has_inventory: Whether inventory is used
        
        Returns:
        - Path to the created Excel file
        """
        print("Creating summary report...")

        # Check if export folder exists
        if not self.export_folder:
            self.initialize_export_folder()
        
        # Create a filename for the summary
        summary_file = os.path.join(self.export_folder, "Summary_Report.xlsx")
        
        # Create summary data
        summary_data = {
            'Category': ['Total Apartments', 'Total Files Created'],
            'Count': [len(apt_files), len(apt_files) + (1 if inv_file else 0)]
        }
        
        # Add direction-specific summaries based on what's available
        if has_pattern:
            # X direction matches
            if 'matched_count_x' in match_results:
                summary_data['Category'].append('X Direction Matches')
                summary_data['Count'].append(match_results['matched_count_x'])
            
            if 'unmatched_count_x' in match_results:
                summary_data['Category'].append('X Direction Unmatched')
                summary_data['Count'].append(match_results['unmatched_count_x'])
            
            # Y direction matches
            if 'matched_count_y' in match_results:
                summary_data['Category'].append('Y Direction Matches')
                summary_data['Count'].append(match_results['matched_count_y'])
            
            if 'unmatched_count_y' in match_results:
                summary_data['Category'].append('Y Direction Unmatched')
                summary_data['Count'].append(match_results['unmatched_count_y'])
        else:
            # All direction matches
            if 'matched_count_all' in match_results:
                summary_data['Category'].append('All Direction Matches')
                summary_data['Count'].append(match_results['matched_count_all'])
            
            if 'unmatched_count_all' in match_results:
                summary_data['Category'].append('All Direction Unmatched')
                summary_data['Count'].append(match_results['unmatched_count_all'])
        
        # Add inventory summaries
        if has_inventory:
            if has_pattern:
                # X inventory
                if 'inv_matched_count_x' in match_results:
                    summary_data['Category'].append('X Direction Inventory Matched')
                    summary_data['Count'].append(match_results['inv_matched_count_x'])
                
                if 'inv_unmatched_count_x' in match_results:
                    summary_data['Category'].append('X Direction Inventory Unmatched')
                    summary_data['Count'].append(match_results['inv_unmatched_count_x'])
                
                # Y inventory
                if 'inv_matched_count_y' in match_results:
                    summary_data['Category'].append('Y Direction Inventory Matched')
                    summary_data['Count'].append(match_results['inv_matched_count_y'])
                
                if 'inv_unmatched_count_y' in match_results:
                    summary_data['Category'].append('Y Direction Inventory Unmatched')
                    summary_data['Count'].append(match_results['inv_unmatched_count_y'])
            else:
                # All inventory
                if 'inv_matched_count_all' in match_results:
                    summary_data['Category'].append('All Direction Inventory Matched')
                    summary_data['Count'].append(match_results['inv_matched_count_all'])
                
                if 'inv_unmatched_count_all' in match_results:
                    summary_data['Category'].append('All Direction Inventory Unmatched')
                    summary_data['Count'].append(match_results['inv_unmatched_count_all'])
        
        # Add totals
        if 'total_matches' in match_results:
            summary_data['Category'].append('Total Matches')
            summary_data['Count'].append(match_results['total_matches'])
        
        if 'total_savings' in match_results:
            summary_data['Category'].append('Total Material Saved (mm²)')
            summary_data['Count'].append(match_results['total_savings'])
            
            summary_data['Category'].append('Total Material Saved (m²)')
            summary_data['Count'].append(match_results['total_savings'] / 1000000)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Create files listing
        files_data = {
            'File Type': ['Summary Report'],
            'Filename': [os.path.basename(summary_file)]
        }
        
        for apt_file in apt_files:
            files_data['File Type'].append(f"Apartment {os.path.basename(apt_file).split('_')[0]}")
            files_data['Filename'].append(os.path.basename(apt_file))
        
        if inv_file:
            files_data['File Type'].append('Inventory Report')
            files_data['Filename'].append(os.path.basename(inv_file))
        
        files_df = pd.DataFrame(files_data)
        
        # Write to Excel
        with pd.ExcelWriter(summary_file, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            worksheet = writer.sheets['Summary']
            worksheet.set_column('A:A', 30)  # Category
            worksheet.set_column('B:B', 15)  # Count
            
            files_df.to_excel(writer, sheet_name='Files', index=False)
            worksheet = writer.sheets['Files']
            worksheet.set_column('A:A', 20)  # File Type
            worksheet.set_column('B:B', 50)  # Filename
        
        print(f"  ✅ Summary report created: {summary_file}")
        return summary_file
    
    def create_zip_file(self):
        """
        Create a zip file containing all reports
        
        Returns:
        - Path to the created ZIP file
        """
        if not self.export_folder:
            return None
            
        # Create zip file name based on export folder name
        base_name = os.path.basename(self.export_folder)
        zip_file = os.path.join(os.path.dirname(self.export_folder), f"{base_name}.zip")
        
        # Create the zip file
        with zipfile.ZipFile(zip_file, 'w') as zipf:
            for root, dirs, files in os.walk(self.export_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(self.export_folder))
                    zipf.write(file_path, arcname)
        
        print(f"✅ Zip file created: {zip_file}")
        return zip_file
    
    def create_apartment_report(self, apt_match_df, inv_match_df, less_df, more_df, direction="X"):
        """
        Create a comprehensive report for apartment matches
        
        Parameters:
        - apt_match_df: DataFrame with apartment matches
        - inv_match_df: DataFrame with inventory matches
        - less_df: DataFrame with unmatched less than half pieces
        - more_df: DataFrame with unmatched more than half pieces
        - direction: Direction of cuts ('X', 'Y', or 'All')
        
        Returns:
        - Dict with raw and styled DataFrames
        """
        import re
        
        # Check if we have data to process
        if (less_df.empty and more_df.empty):
            print(f"Not enough data to create a report for {direction} direction.")
            return None
            
        # Create a mapping of matches by apartment, location, and size
        match_map = {}
        
        # Define match ID prefixes based on direction
        if direction == "X":
            same_apt_prefix = "X"
            diff_apt_prefix = "OX"
            inv_prefix = "IX"
        elif direction == "Y":
            same_apt_prefix = "Y"
            diff_apt_prefix = "OY"
            inv_prefix = "IY"
        else:  # All cuts
            same_apt_prefix = "XY"
            diff_apt_prefix = "O"
            inv_prefix = "I"
        
        # Process apartment matches
        if not apt_match_df.empty:
            for _, match in apt_match_df.iterrows():
                small_apt = match['Small Piece Apt']
                small_loc = match['Small Piece Loc']
                small_size = round(match['Small Piece Size'], 1)
                small_remain = round(match['Small Piece Remain'], 1)
                large_apt = match['Large Piece Apt']
                large_loc = match['Large Piece Loc']
                large_size = round(match['Large Piece Size'], 1)
                match_id = match['Match ID']
                same_apartment = match['Same Apartment']
                
                # Determine the group ID (same as match ID)
                group_id = match_id
                
                # Determine color based on match type
                if same_apartment:
                    # Extract number from match_id (X1, X2, etc.)
                    match_num = int(re.search(r'\d+', match_id).group())
                    color_idx = (match_num - 1) % len(self.color_map['SAME_APT_COLORS'])
                    color = self.color_map['SAME_APT_COLORS'][color_idx]
                else:
                    color = self.color_map['DIFF_APT_COLOR']
                
                # Create keys for small and large pieces
                small_key = (small_apt, small_loc, small_size, 'Less than Half')
                large_key = (large_apt, large_loc, large_size, 'More than Half')
                
                # Create or update match entries for small piece
                if small_key not in match_map:
                    match_map[small_key] = {
                        'match_ids': [], 'group_ids': [], 'colors': [], 
                        'partners': [], 'same_apt': []
                    }
                match_map[small_key]['match_ids'].append(match_id)
                match_map[small_key]['group_ids'].append(group_id)
                match_map[small_key]['colors'].append(color)
                match_map[small_key]['partners'].append((large_apt, large_loc, large_size))
                match_map[small_key]['same_apt'].append(same_apartment)
                
                # Create or update match entries for large piece
                if large_key not in match_map:
                    match_map[large_key] = {
                        'match_ids': [], 'group_ids': [], 'colors': [], 
                        'partners': [], 'same_apt': []
                    }
                match_map[large_key]['match_ids'].append(match_id)
                match_map[large_key]['group_ids'].append(group_id)
                match_map[large_key]['colors'].append(color)
                match_map[large_key]['partners'].append((small_apt, small_loc, small_size))
                match_map[large_key]['same_apt'].append(same_apartment)
        
        # Process inventory matches (for remarks only in apartment report)
        inv_matches_by_apt = {}
        if not inv_match_df.empty:
            for _, match in inv_match_df.iterrows():
                apt = match['Apartment']
                loc = match['Location']
                cut_size = round(match['Cut Size (mm)'], 1)
                cut_remain = round(match['Cut Remain (mm)'], 1)
                inv_loc = match['Inventory Location']
                inv_size = round(match['Inventory Size (mm)'], 1)
                match_id = match['Match ID']
                
                # Create group ID (same as match ID)
                group_id = match_id
                
                # Create key for apartment piece
                apt_key = (apt, loc, cut_size, 'Less than Half')
                
                # Add to inventory matches dictionary
                if apt_key not in inv_matches_by_apt:
                    inv_matches_by_apt[apt_key] = []
                inv_matches_by_apt[apt_key].append((match_id, inv_loc, inv_size, group_id))
        
        # Get all apartment cut pieces
        apartment_pieces = []
        
        # Process less than half pieces
        for _, row in less_df.iterrows():
            apt = row['Apartment']
            loc = row['Location']
            cut_size = round(row['Cut Size (mm)'], 1)
            remain_size = round(row['Remaining Size (mm)'], 1)
            count = int(row['Count'])
            
            # Create key
            key = (apt, loc, cut_size, 'Less than Half')
            
            # Check for apartment matches
            apt_matches = match_map.get(key, {})
            apt_match_ids = apt_matches.get('match_ids', [])
            apt_group_ids = apt_matches.get('group_ids', [])
            apt_colors = apt_matches.get('colors', [])
            apt_partners = apt_matches.get('partners', [])
            apt_same_apt = apt_matches.get('same_apt', [])
            
            # Check for inventory matches
            inv_matches = inv_matches_by_apt.get(key, [])
            
            # If we have apartment matches, create rows for each match
            if apt_match_ids:
                # Group by match partners to consolidate matches
                partner_groups = {}
                for i, partner in enumerate(apt_partners):
                    partner_key = partner
                    if partner_key not in partner_groups:
                        partner_groups[partner_key] = {
                            'ids': [], 'group_ids': [], 'colors': [], 'same_apt': []
                        }
                    if i < len(apt_match_ids):
                        partner_groups[partner_key]['ids'].append(apt_match_ids[i])
                    if i < len(apt_group_ids):
                        partner_groups[partner_key]['group_ids'].append(apt_group_ids[i])
                    if i < len(apt_colors):
                        partner_groups[partner_key]['colors'].append(apt_colors[i])
                    if i < len(apt_same_apt):
                        partner_groups[partner_key]['same_apt'].append(apt_same_apt[i])
                
                # Create a row for each partner group
                for partner, info in partner_groups.items():
                    match_ids = info['ids']
                    group_ids = info['group_ids']
                    colors = info['colors']
                    same_apt_flags = info['same_apt']
                    
                    # They should all have same color and same_apt flag in a group
                    color = colors[0] if colors else ""
                    is_same_apt = same_apt_flags[0] if same_apt_flags else False
                    group_id = group_ids[0] if group_ids else ""
                    
                    # Create remarks
                    partner_apt, partner_loc, _ = partner
                    match_count = len(match_ids)
                    match_ids_str = ', '.join(match_ids)
                    
                    if match_count > 1:
                        remarks = f"{match_count} Matches ({match_ids_str}) with {partner_apt}-{partner_loc} ({'same' if is_same_apt else 'different'} apartment)"
                    else:
                        remarks = f"Match {match_ids_str} with {partner_apt}-{partner_loc} ({'same' if is_same_apt else 'different'} apartment)"
                    
                    # Add inventory match information to remarks if present
                    if inv_matches:
                        inv_locs = {}
                        for match_id, inv_loc, _, _ in inv_matches:
                            if inv_loc not in inv_locs:
                                inv_locs[inv_loc] = []
                            inv_locs[inv_loc].append(match_id)
                        
                        for inv_loc, match_ids in inv_locs.items():
                            match_count = len(match_ids)
                            match_ids_str = ', '.join(match_ids)
                            
                            if match_count > 1:
                                inv_remark = f"{match_count} Inventory Matches ({match_ids_str}) from {inv_loc}"
                            else:
                                inv_remark = f"Inventory Match {match_ids_str} from {inv_loc}"
                            
                            if remarks:
                                remarks += "\n" + inv_remark
                            else:
                                remarks = inv_remark
                    
                    # Create piece entry
                    apartment_pieces.append({
                        'Apartment': apt,
                        'Location': loc,
                        'Cut Size': cut_size,
                        'Remaining Size': remain_size,
                        'Count': len(match_ids),
                        'Match ID': match_ids_str,
                        'Group ID': group_id,
                        'Color': color,
                        'Remarks': remarks,
                        'Piece Type': 'Less than Half',
                        'Match Type': 'Apartment'
                    })
                    
                    # Update count for unmatched pieces
                    count -= len(match_ids)
            
            # If we only have inventory matches (no apartment matches)
            elif inv_matches and not apt_match_ids:
                # Group inventory matches by location
                inv_locs = {}
                for match_id, inv_loc, _, group_id in inv_matches:
                    if inv_loc not in inv_locs:
                        inv_locs[inv_loc] = {
                            'match_ids': [], 'group_ids': []
                        }
                    inv_locs[inv_loc]['match_ids'].append(match_id)
                    inv_locs[inv_loc]['group_ids'].append(group_id)
                
                # Create a row for each inventory location group
                for inv_loc, info in inv_locs.items():
                    match_ids = info['match_ids']
                    group_ids = info['group_ids']
                    
                    # They should all have similar group IDs
                    group_id = group_ids[0] if group_ids else ""
                    
                    match_count = len(match_ids)
                    match_ids_str = ', '.join(match_ids)
                    
                    if match_count > 1:
                        inv_remarks = f"{match_count} Inventory Matches ({match_ids_str}) from {inv_loc}"
                    else:
                        inv_remarks = f"Inventory Match {match_ids_str} from {inv_loc}"
                    
                    # Create piece entry for inventory matches
                    apartment_pieces.append({
                        'Apartment': apt,
                        'Location': loc,
                        'Cut Size': cut_size,
                        'Remaining Size': remain_size,
                        'Count': match_count,
                        'Match ID': match_ids_str,
                        'Group ID': group_id,
                        'Color': self.color_map['INV_COLOR'],  # Fixed color for inventory matches
                        'Remarks': inv_remarks,
                        'Piece Type': 'Less than Half',
                        'Match Type': 'Inventory'  # Changed to Inventory
                    })
                
                # Update count for unmatched pieces
                count -= len(inv_matches)
            
            # Add unmatched pieces if any remain
            if count > 0:
                apartment_pieces.append({
                    'Apartment': apt,
                    'Location': loc,
                    'Cut Size': cut_size,
                    'Remaining Size': remain_size,
                    'Count': count,
                    'Match ID': '',
                    'Group ID': '',
                    'Color': '',
                    'Remarks': f"{count} Unmatched piece(s)",
                    'Piece Type': 'Less than Half',
                    'Match Type': 'Unmatched'
                })
        
        # Process more than half pieces
        for _, row in more_df.iterrows():
            apt = row['Apartment']
            loc = row['Location']
            cut_size = round(row['Cut Size (mm)'], 1)
            remain_size = round(row['Remaining Size (mm)'], 1)
            count = int(row['Count'])
            
            # Create key
            key = (apt, loc, cut_size, 'More than Half')
            
            # Check for apartment matches
            apt_matches = match_map.get(key, {})
            apt_match_ids = apt_matches.get('match_ids', [])
            apt_group_ids = apt_matches.get('group_ids', [])
            apt_colors = apt_matches.get('colors', [])
            apt_partners = apt_matches.get('partners', [])
            apt_same_apt = apt_matches.get('same_apt', [])
            
            # If we have apartment matches, create rows for each match
            if apt_match_ids:
                # Group by match partners to consolidate matches
                partner_groups = {}
                for i, partner in enumerate(apt_partners):
                    partner_key = partner
                    if partner_key not in partner_groups:
                        partner_groups[partner_key] = {
                            'ids': [], 'group_ids': [], 'colors': [], 'same_apt': []
                        }
                    if i < len(apt_match_ids):
                        partner_groups[partner_key]['ids'].append(apt_match_ids[i])
                    if i < len(apt_group_ids):
                        partner_groups[partner_key]['group_ids'].append(apt_group_ids[i])
                    if i < len(apt_colors):
                        partner_groups[partner_key]['colors'].append(apt_colors[i])
                    if i < len(apt_same_apt):
                        partner_groups[partner_key]['same_apt'].append(apt_same_apt[i])
                
                # Create a row for each partner group
                for partner, info in partner_groups.items():
                    match_ids = info['ids']
                    group_ids = info['group_ids']
                    colors = info['colors']
                    same_apt_flags = info['same_apt']
                    
                    # They should all have same color and same_apt flag in a group
                    color = colors[0] if colors else ""
                    is_same_apt = same_apt_flags[0] if same_apt_flags else False
                    group_id = group_ids[0] if group_ids else ""
                    
                    # Create remarks
                    partner_apt, partner_loc, _ = partner
                    match_count = len(match_ids)
                    match_ids_str = ', '.join(match_ids)
                    
                    if match_count > 1:
                        remarks = f"{match_count} Matches ({match_ids_str}) with {partner_apt}-{partner_loc} ({'same' if is_same_apt else 'different'} apartment)"
                    else:
                        remarks = f"Match {match_ids_str} with {partner_apt}-{partner_loc} ({'same' if is_same_apt else 'different'} apartment)"
                    
                    # Create piece entry
                    apartment_pieces.append({
                        'Apartment': apt,
                        'Location': loc,
                        'Cut Size': cut_size,
                        'Remaining Size': remain_size,
                        'Count': len(match_ids),
                        'Match ID': match_ids_str,
                        'Group ID': group_id,
                        'Color': color,
                        'Remarks': remarks,
                        'Piece Type': 'More than Half',
                        'Match Type': 'Apartment'
                    })
                    
                    # Update count for unmatched pieces
                    count -= len(match_ids)
            
            # Add unmatched pieces if any remain
            if count > 0:
                apartment_pieces.append({
                    'Apartment': apt,
                    'Location': loc,
                    'Cut Size': cut_size,
                    'Remaining Size': remain_size,
                    'Count': count,
                    'Match ID': '',
                    'Group ID': '',
                    'Color': '',
                    'Remarks': f"{count} Unmatched piece(s)",
                    'Piece Type': 'More than Half',
                    'Match Type': 'Unmatched'
                })
        
        # Create DataFrame from pieces
        apartment_df = pd.DataFrame(apartment_pieces)
        
        # If DataFrame is empty, create an empty one with correct columns
        if apartment_df.empty:
            apartment_df = pd.DataFrame(columns=[
                'Apartment', 'Location', 'Cut Size', 'Remaining Size', 'Count',
                'Match ID', 'Group ID', 'Color', 'Remarks', 'Piece Type', 'Match Type'
            ])
        
        # Sort by Apartment, Cut Size
        apartment_df = apartment_df.sort_values(by=['Match Type', 'Apartment', 'Cut Size'])
        apartment_df.reset_index(drop=True, inplace=True)
        
        return {
            'raw_df': apartment_df,
            'direction': direction
        }
    
    def create_inventory_report(self, inv_match_df, direction="X"):
        """
        Create an inventory report showing matched and unmatched pieces
        
        Parameters:
        - inv_match_df: DataFrame with inventory matches
        - direction: Direction of cuts ('X', 'Y', or 'All')
        
        Returns:
        - DataFrame with formatted inventory data
        """
        # Check if there's any data to process
        if inv_match_df.empty:
            return pd.DataFrame()
            
        # Create inventory report rows
        inventory_rows = []
        
        # Process inventory matches
        for _, match in inv_match_df.iterrows():
            inv_loc = match['Inventory Location']
            inv_size = round(match['Inventory Size (mm)'], 1)
            apt = match['Apartment']
            loc = match['Location']
            cut_size = round(match['Cut Size (mm)'], 1)
            waste = round(match['Waste'], 1)
            match_id = match['Match ID']
            
            # Group ID is the same as match ID
            group_id = match_id
            
            # Determine if this piece is matched or unmatched
            status = 'Matched'
            
            # Add to inventory rows
            inventory_rows.append({
                'Location': inv_loc,
                'Size (mm)': inv_size,
                'Match ID': match_id,
                'Group ID': group_id,
                'Status': status,
                'Matched With': f"{apt}-{loc} ({cut_size}mm)",
                'Cut Size (mm)': cut_size,
                'Waste (mm)': waste,
                'Piece Type': 'Inventory',
                'Color': self.color_map['INV_COLOR'],
                'Direction': direction
            })
        
        # Create DataFrame
        inventory_df = pd.DataFrame(inventory_rows)
        
        # Sort by Location, Size, and Status (Matched first)
        if not inventory_df.empty:
            # Create a custom sorter for Status
            inventory_df['Status_Order'] = inventory_df['Status'].map({'Matched': 0, 'Unmatched': 1})
            inventory_df = inventory_df.sort_values(by=['Location', 'Size (mm)', 'Status_Order', 'Match ID'])
            inventory_df = inventory_df.drop(columns=['Status_Order'])
            inventory_df.reset_index(drop=True, inplace=True)
        
        return inventory_df
    
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
                    'Total Material Saved (mm²)',
                    'Total Material Saved (m²)',
                    'Optimization Date'
                ],
                'Value': [
                    optimization_results.get('matched_count', 0),
                    optimization_results.get('within_apartment_matches', 0),
                    optimization_results.get('cross_apartment_matches', 0),
                    optimization_results.get('total_savings', 0),
                    optimization_results.get('total_savings', 0) / 1000000,
                    timestamp
                ]
            }
            
            # Add inventory matches if available
            if 'inventory_matches' in optimization_results:
                summary_data['Metric'].append('Inventory Matches')
                summary_data['Value'].append(optimization_results['inventory_matches'])
            
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
                    
                    if not isinstance(x_inventory, pd.DataFrame):
                        x_inventory = pd.DataFrame(x_inventory) if x_inventory else pd.DataFrame()
                    
                    if not isinstance(y_inventory, pd.DataFrame):
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
                    
                    if not isinstance(all_inventory, pd.DataFrame):
                        all_inventory = pd.DataFrame(all_inventory) if all_inventory else pd.DataFrame()
                    
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

    def export_matching_results_to_excel(self, match_results):
        """
        Export matching results to an Excel file with improved formatting and colored cells
        
        Parameters:
        -----------
        match_results : dict
            Dictionary with match results data
            
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
        
        # Create a timestamp string for the report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Extract data from match_results
        has_pattern = match_results.get('has_pattern', True)
        
        # Get simplified matches for a unified match list
        simplified_matches = match_results.get('simplified_matches', [])
        if not simplified_matches:
            simplified_matches = []
        
        # Convert to DataFrame if it's not already
        if not isinstance(simplified_matches, pd.DataFrame):
            simplified_matches_df = pd.DataFrame(simplified_matches) if simplified_matches else pd.DataFrame()
        else:
            simplified_matches_df = simplified_matches
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Create summary sheet
            summary_data = {
                'Metric': [
                    'Total Matches', 
                    'Within-Apartment Matches', 
                    'Cross-Apartment Matches', 
                    'Inventory Matches',
                    'Material Saved (mm²)',
                    'Material Saved (m²)',
                    'Report Date'
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
            
            # Add header format
            header_format = writer.book.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            # Add title format
            title_format = writer.book.add_format({
                'bold': True,
                'font_size': 14,
                'border': 0
            })
            
            # Add formats for match types
            same_apt_format = writer.book.add_format({'bg_color': '#FFD700', 'border': 1})
            cross_apt_format = writer.book.add_format({'bg_color': '#A9A9A9', 'border': 1})
            inv_format = writer.book.add_format({'bg_color': '#8FBC8F', 'border': 1})
            
            # Format the summary sheet
            summary_sheet = writer.sheets['Summary']
            summary_sheet.set_column('A:A', 25)
            summary_sheet.set_column('B:B', 15)
            
            # Add title to summary sheet
            summary_sheet.write(0, 0, 'Tile Cut Piece Optimization Report', title_format)
            
            # Process simplified matches for "All Matches" sheet
            if not simplified_matches_df.empty:
                # Apply proper column names if needed
                if 'Match ID' not in simplified_matches_df.columns and 'match_id' in simplified_matches_df.columns:
                    simplified_matches_df = simplified_matches_df.rename(columns={
                        'match_id': 'Match ID',
                        'from': 'From',
                        'to': 'To',
                        'size': 'Size (mm)',
                        'match_type': 'Match Type'
                    })
                
                # Write to match details sheet
                simplified_matches_df.to_excel(writer, sheet_name='Match Details', index=False)
                match_sheet = writer.sheets['Match Details']
                
                # Format the match details sheet headers
                for col_num, col_name in enumerate(simplified_matches_df.columns):
                    match_sheet.write(0, col_num, col_name, header_format)
                    match_sheet.set_column(col_num, col_num, 15)
                
                # Add conditional formatting for match types
                if 'Match Type' in simplified_matches_df.columns:
                    # Format for Same Apartment matches
                    match_sheet.conditional_format(1, 0, len(simplified_matches_df) + 1, len(simplified_matches_df.columns) - 1, {
                        'type': 'formula',
                        'criteria': '=$F2="Same Apartment"',
                        'format': same_apt_format
                    })
                    
                    # Format for Cross Apartment matches
                    match_sheet.conditional_format(1, 0, len(simplified_matches_df) + 1, len(simplified_matches_df.columns) - 1, {
                        'type': 'formula',
                        'criteria': '=$F2="Cross Apartment"',
                        'format': cross_apt_format
                    })
                    
                    # Format for Inventory matches
                    match_sheet.conditional_format(1, 0, len(simplified_matches_df) + 1, len(simplified_matches_df.columns) - 1, {
                        'type': 'formula',
                        'criteria': '=$F2="Inventory"',
                        'format': inv_format
                    })
            
            # Process pattern-specific data if available
            if has_pattern:
                # X matches
                x_matches = [m for m in simplified_matches if (isinstance(m, dict) and m.get('Match ID', '').startswith(('X', 'IX', 'OX')))]
                if x_matches:
                    x_df = pd.DataFrame(x_matches)
                    x_df.to_excel(writer, sheet_name='X Matches', index=False)
                    
                    x_sheet = writer.sheets['X Matches']
                    for col_num, col_name in enumerate(x_df.columns):
                        x_sheet.write(0, col_num, col_name, header_format)
                        x_sheet.set_column(col_num, col_num, 15)
                        
                    # Add conditional formatting for X matches
                    if 'Match Type' in x_df.columns:
                        x_sheet.conditional_format(1, 0, len(x_df) + 1, len(x_df.columns) - 1, {
                            'type': 'cell',
                            'criteria': 'equal to',
                            'value': '"Same Apartment"',
                            'format': same_apt_format
                        })
                        x_sheet.conditional_format(1, 0, len(x_df) + 1, len(x_df.columns) - 1, {
                            'type': 'cell',
                            'criteria': 'equal to',
                            'value': '"Cross Apartment"',
                            'format': cross_apt_format
                        })
                        x_sheet.conditional_format(1, 0, len(x_df) + 1, len(x_df.columns) - 1, {
                            'type': 'cell',
                            'criteria': 'equal to',
                            'value': '"Inventory"',
                            'format': inv_format
                        })
                
                # Y matches
                y_matches = [m for m in simplified_matches if (isinstance(m, dict) and m.get('Match ID', '').startswith(('Y', 'IY', 'OY')))]
                if y_matches:
                    y_df = pd.DataFrame(y_matches)
                    y_df.to_excel(writer, sheet_name='Y Matches', index=False)
                    
                    y_sheet = writer.sheets['Y Matches']
                    for col_num, col_name in enumerate(y_df.columns):
                        y_sheet.write(0, col_num, col_name, header_format)
                        y_sheet.set_column(col_num, col_num, 15)
                        
                    # Add conditional formatting for Y matches
                    if 'Match Type' in y_df.columns:
                        y_sheet.conditional_format(1, 0, len(y_df) + 1, len(y_df.columns) - 1, {
                            'type': 'cell',
                            'criteria': 'equal to',
                            'value': '"Same Apartment"',
                            'format': same_apt_format
                        })
                        y_sheet.conditional_format(1, 0, len(y_df) + 1, len(y_df.columns) - 1, {
                            'type': 'cell',
                            'criteria': 'equal to',
                            'value': '"Cross Apartment"',
                            'format': cross_apt_format
                        })
                        y_sheet.conditional_format(1, 0, len(y_df) + 1, len(y_df.columns) - 1, {
                            'type': 'cell',
                            'criteria': 'equal to',
                            'value': '"Inventory"',
                            'format': inv_format
                        })
                
                # Add unmatched pieces sheets if available
                for key, sheet_name in [
                    ('x_unmatched_less', 'X Unmatched < Half'),
                    ('x_unmatched_more', 'X Unmatched > Half'),
                    ('y_unmatched_less', 'Y Unmatched < Half'),
                    ('y_unmatched_more', 'Y Unmatched > Half')
                ]:
                    if key in match_results:
                        unmatched_data = match_results[key]
                        
                        # Convert to DataFrame if it's a list
                        if isinstance(unmatched_data, list) and unmatched_data:
                            unmatched_df = pd.DataFrame(unmatched_data)
                        elif isinstance(unmatched_data, pd.DataFrame) and not unmatched_data.empty:
                            unmatched_df = unmatched_data
                        else:
                            continue
                        
                        # Write to Excel with proper column names
                        unmatched_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Excel limits sheet names to 31 chars
                        
                        # Format sheet
                        sheet = writer.sheets[sheet_name[:31]]
                        for col_num, col_name in enumerate(unmatched_df.columns):
                            sheet.write(0, col_num, col_name, header_format)
                            sheet.set_column(col_num, col_num, 15)
            else:
                # For non-pattern mode, create "All Matches" sheet
                simplified_matches_df.to_excel(writer, sheet_name='All Matches', index=False)
                
                all_sheet = writer.sheets['All Matches']
                for col_num, col_name in enumerate(simplified_matches_df.columns):
                    all_sheet.write(0, col_num, col_name, header_format)
                    all_sheet.set_column(col_num, col_num, 15)
                
                # Add unmatched pieces sheets if available
                for key, sheet_name in [
                    ('all_unmatched_less', 'Unmatched < Half'),
                    ('all_unmatched_more', 'Unmatched > Half')
                ]:
                    if key in match_results:
                        unmatched_data = match_results[key]
                        
                        # Convert to DataFrame if it's a list
                        if isinstance(unmatched_data, list) and unmatched_data:
                            unmatched_df = pd.DataFrame(unmatched_data)
                        elif isinstance(unmatched_data, pd.DataFrame) and not unmatched_data.empty:
                            unmatched_df = unmatched_data
                        else:
                            continue
                        
                        # Write to Excel
                        unmatched_df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # Format sheet
                        sheet = writer.sheets[sheet_name]
                        for col_num, col_name in enumerate(unmatched_df.columns):
                            sheet.write(0, col_num, col_name, header_format)
                            sheet.set_column(col_num, col_num, 15)
            
            # Create instructions sheet
            instructions_data = {
                'Instructions': [
                    'TILE CUT PIECE OPTIMIZATION REPORT',
                    '-----------------------------------',
                    '',
                    'This report contains detailed information about matched and unmatched tiles from the optimization process.',
                    '',
                    'SHEETS IN THIS REPORT:',
                    '1. Summary: Overall statistics about the optimization process',
                    '2. Match Details: Complete list of all matched cut pieces',
                    '3. X/Y Matches: Detailed listings of matches by direction',
                    '4. Unmatched Pieces: Lists of tiles that could not be matched',
                    '',
                    'MATCH TYPES:',
                    '- Same Apartment Match: Pieces from the same apartment (yellow)',
                    '- Cross Apartment Match: Pieces from different apartments (gray)',
                    '- Inventory Match: Matches with inventory pieces (green)',
                    '',
                    'TO IMPLEMENT THESE MATCHES:',
                    '1. Use the remaining pieces from cut tiles as indicated in the matches',
                    '2. Measure carefully to ensure proper fit',
                    '3. Tag pieces according to their Match IDs for easy identification',
                    '',
                    f'Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                ]
            }
            
            instructions_df = pd.DataFrame(instructions_data)
            instructions_df.to_excel(writer, sheet_name='Instructions', index=False)
            
            # Format the instructions sheet
            instructions_sheet = writer.sheets['Instructions']
            instructions_sheet.set_column('A:A', 100)
            
            # Add title to instructions sheet
            instructions_sheet.write(0, 0, 'INSTRUCTIONS', title_format)
        
        # Reset the buffer position
        output.seek(0)
        
        return output

    def format_simplified_matches(self, match_results):
        """
        Create a simplified dataframe of all matches for easy export
        
        Parameters:
        -----------
        match_results : dict
            Dictionary with match results from optimization
            
        Returns:
        --------
        DataFrame
            Simplified matches DataFrame
        """
        import pandas as pd
        
        has_pattern = match_results.get('has_pattern', True)
        simplified_matches = []
        
        # Process based on pattern mode
        if has_pattern:
            # X direction apartment matches
            if 'x_apt_matches' in match_results and not match_results['x_apt_matches'].empty:
                for _, match in match_results['x_apt_matches'].iterrows():
                    match_type = "Same Apartment" if match['Same Apartment'] else "Cross Apartment"
                    simplified_matches.append({
                        'Match ID': match['Match ID'],
                        'From': f"{match['Small Piece Apt']}-{match['Small Piece Loc']}",
                        'To': f"{match['Large Piece Apt']}-{match['Large Piece Loc']}",
                        'Size (mm)': round(match['Small Piece Size'], 1),
                        'Waste (mm)': round(match['Waste'], 1),
                        'Match Type': match_type
                    })
            
            # Y direction apartment matches
            if 'y_apt_matches' in match_results and not match_results['y_apt_matches'].empty:
                for _, match in match_results['y_apt_matches'].iterrows():
                    match_type = "Same Apartment" if match['Same Apartment'] else "Cross Apartment"
                    simplified_matches.append({
                        'Match ID': match['Match ID'],
                        'From': f"{match['Small Piece Apt']}-{match['Small Piece Loc']}",
                        'To': f"{match['Large Piece Apt']}-{match['Large Piece Loc']}",
                        'Size (mm)': round(match['Small Piece Size'], 1),
                        'Waste (mm)': round(match['Waste'], 1),
                        'Match Type': match_type
                    })
            
            # X inventory matches
            if 'x_inv_matches' in match_results and not match_results['x_inv_matches'].empty:
                for _, match in match_results['x_inv_matches'].iterrows():
                    simplified_matches.append({
                        'Match ID': match['Match ID'],
                        'From': f"{match['Apartment']}-{match['Location']}",
                        'To': f"Inventory-{match['Inventory Location']}",
                        'Size (mm)': round(match['Cut Size (mm)'], 1),
                        'Waste (mm)': round(match['Waste'], 1),
                        'Match Type': "Inventory"
                    })
            
            # Y inventory matches
            if 'y_inv_matches' in match_results and not match_results['y_inv_matches'].empty:
                for _, match in match_results['y_inv_matches'].iterrows():
                    simplified_matches.append({
                        'Match ID': match['Match ID'],
                        'From': f"{match['Apartment']}-{match['Location']}",
                        'To': f"Inventory-{match['Inventory Location']}",
                        'Size (mm)': round(match['Cut Size (mm)'], 1),
                        'Waste (mm)': round(match['Waste'], 1),
                        'Match Type': "Inventory"
                    })
        else:
            # All cut apartment matches
            if 'all_apt_matches' in match_results and not match_results['all_apt_matches'].empty:
                for _, match in match_results['all_apt_matches'].iterrows():
                    match_type = "Same Apartment" if match['Same Apartment'] else "Cross Apartment"
                    simplified_matches.append({
                        'Match ID': match['Match ID'],
                        'From': f"{match['Small Piece Apt']}-{match['Small Piece Loc']}",
                        'To': f"{match['Large Piece Apt']}-{match['Large Piece Loc']}",
                        'Size (mm)': round(match['Small Piece Size'], 1),
                        'Waste (mm)': round(match['Waste'], 1),
                        'Match Type': match_type
                    })
            
            # All cut inventory matches
            if 'all_inv_matches' in match_results and not match_results['all_inv_matches'].empty:
                for _, match in match_results['all_inv_matches'].iterrows():
                    simplified_matches.append({
                        'Match ID': match['Match ID'],
                        'From': f"{match['Apartment']}-{match['Location']}",
                        'To': f"Inventory-{match['Inventory Location']}",
                        'Size (mm)': round(match['Cut Size (mm)'], 1),
                        'Waste (mm)': round(match['Waste'], 1),
                        'Match Type': "Inventory"
                    })
        
        # Create DataFrame and sort by Match ID
        if simplified_matches:
            simplified_df = pd.DataFrame(simplified_matches)
            return simplified_df.sort_values('Match ID').reset_index(drop=True)
        else:
            return pd.DataFrame(columns=['Match ID', 'From', 'To', 'Size (mm)', 'Waste (mm)', 'Match Type'])

    def download_full_report(self, match_results, inventory_data=None):
        """
        Generate a complete report package with all details
        
        Parameters:
        -----------
        match_results : dict
            Dictionary with match results data
        inventory_data : dict, optional
            Dictionary with inventory data if available
            
        Returns:
        --------
        BytesIO
            ZIP file as a BytesIO object
        """
        import io
        import zipfile
        import pandas as pd
        from datetime import datetime
        import os
        import tempfile
        import matplotlib.pyplot as plt
        import base64
        
        # Create a timestamp string for the report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create temporary directory for files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a zip file in memory
            zip_buffer = io.BytesIO()
            
            # Create the basic reports
            match_report_data = self.export_matching_results_to_excel(match_results)
            
            # Save the match report
            match_report_path = os.path.join(temp_dir, f"Matching_Results_{timestamp}.xlsx")
            with open(match_report_path, 'wb') as f:
                f.write(match_report_data.getvalue())
            
            # Create visualization report if available
            visualization_path = None
            if 'optimization_plot' in match_results and match_results['optimization_plot']:
                # Decode base64 image
                img_data = base64.b64decode(match_results['optimization_plot'])
                visualization_path = os.path.join(temp_dir, f"Optimization_Visualization_{timestamp}.png")
                with open(visualization_path, 'wb') as f:
                    f.write(img_data)
            
            # Generate apartment-specific reports if needed
            apartment_files = []
            has_pattern = match_results.get('has_pattern', True)
            
            # Extract apartments from match data
            apartments = set()
            simplified_matches = match_results.get('simplified_matches', [])
            for match in simplified_matches:
                if 'From' in match and '-' in match['From']:
                    apartment = match['From'].split('-')[0]
                    apartments.add(apartment)
                if 'To' in match and '-' in match['To']:
                    apartment = match['To'].split('-')[0]
                    apartments.add(apartment)
            
            # Create apartment reports
            for apartment in apartments:
                # Filter matches for this apartment
                apt_matches = [m for m in simplified_matches if 
                            ('From' in m and m['From'].startswith(apartment + '-')) or 
                            ('To' in m and m['To'].startswith(apartment + '-'))]
                
                if apt_matches:
                    # Create a DataFrame
                    apt_df = pd.DataFrame(apt_matches)
                    
                    # Create an Excel file
                    apt_buffer = io.BytesIO()
                    with pd.ExcelWriter(apt_buffer, engine='xlsxwriter') as writer:
                        # Add apartment match summary
                        apt_df.to_excel(writer, sheet_name='Matches', index=False)
                        
                        # Format the worksheet
                        worksheet = writer.sheets['Matches']
                        for col_num, col_name in enumerate(apt_df.columns):
                            worksheet.write(0, col_num, col_name, writer.book.add_format({
                                'bold': True, 'fg_color': '#D7E4BC', 'border': 1
                            }))
                            worksheet.set_column(col_num, col_num, 15)
                        
                        # Split matches by type and add to separate sheets
                        same_apt_matches = [m for m in apt_matches if 
                                            m.get('Match Type') == 'Same Apartment' or 
                                            m.get('Match ID', '').startswith(('X', 'Y'))]
                        cross_apt_matches = [m for m in apt_matches if 
                                            m.get('Match Type') == 'Cross Apartment' or 
                                            m.get('Match ID', '').startswith(('OX', 'OY'))]
                        inv_matches = [m for m in apt_matches if 
                                    m.get('Match Type') == 'Inventory' or 
                                    m.get('Match ID', '').startswith(('IX', 'IY'))]
                        
                        # Add sheets if we have data
                        if same_apt_matches:
                            pd.DataFrame(same_apt_matches).to_excel(writer, sheet_name='Same Apt Matches', index=False)
                        if cross_apt_matches:
                            pd.DataFrame(cross_apt_matches).to_excel(writer, sheet_name='Cross Apt Matches', index=False)
                        if inv_matches:
                            pd.DataFrame(inv_matches).to_excel(writer, sheet_name='Inventory Matches', index=False)
                    
                    # Save the apartment report
                    apt_buffer.seek(0)
                    apt_file_path = os.path.join(temp_dir, f"{apartment}_Matches_{timestamp}.xlsx")
                    with open(apt_file_path, 'wb') as f:
                        f.write(apt_buffer.getvalue())
                    apartment_files.append(apt_file_path)
            
            # Generate inventory report if inventory data is available
            inventory_report_path = None
            if inventory_data and 'has_pattern' in inventory_data:
                inv_buffer = io.BytesIO()
                with pd.ExcelWriter(inv_buffer, engine='xlsxwriter') as writer:
                    # Write inventory data based on pattern mode
                    has_pattern_inv = inventory_data.get('has_pattern', True)
                    
                    if has_pattern_inv:
                        # X inventory
                        x_inv = inventory_data.get('cut_x_inventory', [])
                        if x_inv:
                            if isinstance(x_inv, list):
                                x_inv_df = pd.DataFrame(x_inv)
                            else:
                                x_inv_df = x_inv
                            x_inv_df.to_excel(writer, sheet_name='X Inventory', index=False)
                        
                        # Y inventory
                        y_inv = inventory_data.get('cut_y_inventory', [])
                        if y_inv:
                            if isinstance(y_inv, list):
                                y_inv_df = pd.DataFrame(y_inv)
                            else:
                                y_inv_df = y_inv
                            y_inv_df.to_excel(writer, sheet_name='Y Inventory', index=False)
                    else:
                        # All inventory
                        all_inv = inventory_data.get('all_cut_inventory', [])
                        if all_inv:
                            if isinstance(all_inv, list):
                                all_inv_df = pd.DataFrame(all_inv)
                            else:
                                all_inv_df = all_inv
                            all_inv_df.to_excel(writer, sheet_name='All Inventory', index=False)
                    
                    # Add summary sheet
                    summary_data = {
                        'Inventory Summary': [
                            f'Pattern mode: {"Yes" if has_pattern_inv else "No"}',
                            f'Total inventory pieces: {sum(len(inventory_data.get(k, [])) for k in ["cut_x_inventory", "cut_y_inventory", "all_cut_inventory"])}',
                            f'Inventory matches used: {match_results.get("inventory_matches", 0)}',
                            f'Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                        ]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Save the inventory report
                inv_buffer.seek(0)
                inventory_report_path = os.path.join(temp_dir, f"Inventory_Report_{timestamp}.xlsx")
                with open(inventory_report_path, 'wb') as f:
                    f.write(inv_buffer.getvalue())
            
            # Generate a summary text file
            summary_path = os.path.join(temp_dir, "README.txt")
            with open(summary_path, 'w') as f:
                f.write("TILE OPTIMIZATION REPORT PACKAGE\n")
                f.write("=============================\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Total Matches: {match_results.get('matched_count', 0)}\n")
                f.write(f"Within-Apartment Matches: {match_results.get('within_apartment_matches', 0)}\n")
                f.write(f"Cross-Apartment Matches: {match_results.get('cross_apartment_matches', 0)}\n")
                f.write(f"Inventory Matches: {match_results.get('inventory_matches', 0)}\n")
                f.write(f"Material Saved: {match_results.get('total_savings', 0)} mm² ")
                f.write(f"({match_results.get('total_savings', 0)/1000000:.2f} m²)\n\n")
                f.write("Files included in this package:\n")
                f.write("- Matching_Results.xlsx: Complete matching results\n")
                if visualization_path:
                    f.write("- Optimization_Visualization.png: Visual representation of matches\n")
                if apartment_files:
                    f.write(f"- Apartment Reports: {len(apartment_files)} apartment-specific match reports\n")
                if inventory_report_path:
                    f.write("- Inventory_Report.xlsx: Inventory status after matching\n")
                f.write("\nTo use these results in your project:\n")
                f.write("1. Review the matches in the Excel file\n")
                f.write("2. Verify all matches before implementation\n")
                f.write("3. For detailed instructions, see the Instructions sheet in the Excel file\n")
            
            # Create the ZIP file with all reports
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add match report
                zipf.write(match_report_path, f"Matching_Results.xlsx")
                
                # Add visualization if available
                if visualization_path:
                    zipf.write(visualization_path, f"Optimization_Visualization.png")
                
                # Add apartment reports
                for apt_file in apartment_files:
                    zipf.write(apt_file, os.path.basename(apt_file))
                
                # Add inventory report if available
                if inventory_report_path:
                    zipf.write(inventory_report_path, f"Inventory_Report.xlsx")
                
                # Add summary text file
                zipf.write(summary_path, "README.txt")
            
            # Reset the buffer position
            zip_buffer.seek(0)
            
            return zip_buffer

    def generate_match_visualization(self, match_results, room_df):
        """
        Generate a visualization showing matched cut pieces
        
        Parameters:
        -----------
        match_results : dict
            Dictionary with match results data
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
        import pandas as pd
        
        plt.figure(figsize=(16, 12))
        
        # Define colors for different match types
        SAME_APT_COLOR = '#FFD700'  # Gold
        DIFF_APT_COLOR = '#FF6347'  # Tomato
        INV_COLOR = '#00BFFF'       # Deep Sky Blue
        
        # Check if room_df contains polygon information
        has_polygons = False
        if isinstance(room_df, pd.DataFrame) and 'polygon' in room_df.columns:
            has_polygons = True
        
        # Plot room boundaries if polygons are available
        if has_polygons:
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
        
        # Process simplified matches for visualization
        simplified_matches = match_results.get('simplified_matches', [])
        
        # Create a mapping from match ID to color
        match_colors = {}
        same_apt_match_count = 0
        
        for match in simplified_matches:
            match_id = match.get('Match ID', '')
            match_type = None
            
            # Determine match type
            if match_id.startswith('I') or match_id.startswith('IX') or match_id.startswith('IY'):
                match_type = 'inventory'
                match_colors[match_id] = INV_COLOR
            elif match_id.startswith('O') or match_id.startswith('OX') or match_id.startswith('OY'):
                match_type = 'cross_apt'
                match_colors[match_id] = DIFF_APT_COLOR
            else:
                match_type = 'same_apt'
                same_apt_match_count += 1
                
                # Use color cycling for same apartment matches
                color_idx = (same_apt_match_count - 1) % len(self.color_map['SAME_APT_COLORS'])
                match_colors[match_id] = self.color_map['SAME_APT_COLORS'][color_idx]
                
            # Draw line between matched pieces
            if 'From' in match and 'To' in match:
                # These would be labeled points on the plot
                # In a full implementation, you would calculate the actual coordinates
                # For now, we'll use placeholder coordinates
                x1, y1 = 100 + same_apt_match_count * 10, 100 + same_apt_match_count * 5
                x2, y2 = 150 + same_apt_match_count * 10, 150 + same_apt_match_count * 5
                
                plt.plot([x1, x2], [y1, y2], color=match_colors[match_id], linewidth=2)
                
                # Add match labels
                plt.text(x1, y1, match['From'], fontsize=8, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))
                plt.text(x2, y2, match['To'], fontsize=8, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))
                
                # Add match ID at midpoint
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                plt.text(mid_x, mid_y, match_id, fontsize=8, ha='center', va='center',
                        bbox=dict(facecolor=match_colors[match_id], alpha=0.9, boxstyle='round,pad=0.1'))
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=SAME_APT_COLOR, alpha=0.7, edgecolor='black', label='Same Apartment Match'),
            Patch(facecolor=DIFF_APT_COLOR, alpha=0.7, edgecolor='black', label='Cross Apartment Match'),
            Patch(facecolor=INV_COLOR, alpha=0.7, edgecolor='black', label='Inventory Match')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add stats in a text box
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

    def export_full_report(self, match_results, has_pattern=True, has_inventory=False):
        """
        Export a full detailed report of all matching results
        
        Parameters:
        - match_results: Dictionary with match results
        - has_pattern: Whether pattern mode is used (X/Y cuts)
        - has_inventory: Whether inventory is used
        
        Returns:
        - Dictionary with paths to created files
        """
        print("\n📊 Exporting full detailed report...")
        
        # Initialize export folder
        export_folder = self.initialize_export_folder()
        
        # Get apartment raw data from the match results
        if has_pattern:
            # Process for X direction
            x_apt_df = None
            x_inv_df = None
            
            if 'x_report' in match_results:
                x_apt_df = match_results['x_report']['raw_df']
            
            if 'x_inv_report' in match_results:
                x_inv_df = match_results['x_inv_report']
            
            # Process for Y direction
            y_apt_df = None
            y_inv_df = None
            
            if 'y_report' in match_results:
                y_apt_df = match_results['y_report']['raw_df']
            
            if 'y_inv_report' in match_results:
                y_inv_df = match_results['y_inv_report']
            
            # Combine X and Y for apartment data
            apt_raw_df = pd.DataFrame()
            if x_apt_df is not None and not x_apt_df.empty:
                apt_raw_df = pd.concat([apt_raw_df, x_apt_df])
            
            if y_apt_df is not None and not y_apt_df.empty:
                apt_raw_df = pd.concat([apt_raw_df, y_apt_df])
            
            # Combine X and Y for inventory data
            inv_df = pd.DataFrame()
            if x_inv_df is not None and not x_inv_df.empty:
                inv_df = pd.concat([inv_df, x_inv_df])
            
            if y_inv_df is not None and not y_inv_df.empty:
                inv_df = pd.concat([inv_df, y_inv_df])
        else:
            # Process for All direction
            apt_raw_df = match_results.get('all_report', {}).get('raw_df', pd.DataFrame())
            inv_df = match_results.get('all_inv_report', pd.DataFrame())
        
        # Get unique apartments
        apartments = []
        if not apt_raw_df.empty and 'Apartment' in apt_raw_df.columns:
            apartments = sorted(apt_raw_df['Apartment'].unique())
        
        print(f"Found {len(apartments)} apartments to process")
        
        # Process each apartment
        apt_files = []
        for apt in apartments:
            apt_file = self.export_apartment_data(apt_raw_df, apt, has_pattern=has_pattern)
            if apt_file:
                apt_files.append(apt_file)
        
        # Process inventory
        inv_file = None
        if has_inventory and (not inv_df.empty or not apt_raw_df.empty):
            inv_file = self.export_inventory_data(inv_df, apt_raw_df, has_pattern=has_pattern)
        
        # Create match count summaries
        count_summary = {}
        
        # Add direction-specific counts
        if has_pattern:
            # X direction counts
            if x_apt_df is not None and not x_apt_df.empty:
                count_summary['matched_count_x'] = len(x_apt_df[x_apt_df['Match Type'] == 'Apartment'])
                count_summary['unmatched_count_x'] = len(x_apt_df[x_apt_df['Match Type'] == 'Unmatched'])
            
            # Y direction counts
            if y_apt_df is not None and not y_apt_df.empty:
                count_summary['matched_count_y'] = len(y_apt_df[y_apt_df['Match Type'] == 'Apartment'])
                count_summary['unmatched_count_y'] = len(y_apt_df[y_apt_df['Match Type'] == 'Unmatched'])
            
            # X inventory counts
            if x_inv_df is not None and not x_inv_df.empty and 'Status' in x_inv_df.columns:
                count_summary['inv_matched_count_x'] = len(x_inv_df[x_inv_df['Status'] == 'Matched'])
                count_summary['inv_unmatched_count_x'] = len(x_inv_df[x_inv_df['Status'] == 'Unmatched'])
            
            # Y inventory counts
            if y_inv_df is not None and not y_inv_df.empty and 'Status' in y_inv_df.columns:
                count_summary['inv_matched_count_y'] = len(y_inv_df[y_inv_df['Status'] == 'Matched'])
                count_summary['inv_unmatched_count_y'] = len(y_inv_df[y_inv_df['Status'] == 'Unmatched'])
        else:
            # All direction counts
            if not apt_raw_df.empty and 'Match Type' in apt_raw_df.columns:
                count_summary['matched_count_all'] = len(apt_raw_df[apt_raw_df['Match Type'] == 'Apartment'])
                count_summary['unmatched_count_all'] = len(apt_raw_df[apt_raw_df['Match Type'] == 'Unmatched'])
            
            # All inventory counts
            if not inv_df.empty and 'Status' in inv_df.columns:
                count_summary['inv_matched_count_all'] = len(inv_df[inv_df['Status'] == 'Matched'])
                count_summary['inv_unmatched_count_all'] = len(inv_df[inv_df['Status'] == 'Unmatched'])
        
        # Add totals
        count_summary['total_matches'] = match_results.get('matched_count', 0)
        count_summary['total_savings'] = match_results.get('total_savings', 0)
        
        # Create summary report
        summary_file = self.create_summary_report(apt_files, inv_file, count_summary, has_pattern, has_inventory)
        
        # Create zip file of all reports
        zip_file = self.create_zip_file()
        
        return {
            'export_folder': export_folder,
            'apt_files': apt_files,
            'inv_file': inv_file,
            'summary_file': summary_file,
            'zip_file': zip_file
        }