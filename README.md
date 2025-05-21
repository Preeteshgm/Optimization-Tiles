Tile Layout Web Application
A web-based application for processing DXF files to create optimized tile layouts with material wastage analysis.

Overview
This application transforms the tile layout process from a complex manual task into a streamlined workflow. It processes DXF files containing room boundaries and start points, clusters rooms into apartments, applies tile layouts with proper grout spacing, classifies tiles by type, identifies problematic small cuts, and provides comprehensive reports with wastage analysis.

Features
DXF Processing: Extract room boundaries and start points from DXF files
Room Clustering: Automatically group rooms into apartments
Custom Naming: Assign meaningful names to apartments and rooms
Orientation Control: Set apartment orientations for proper tile alignment
Tile Coverage: Generate grid-aligned tiles with explicit grout spacing
Tile Classification: Classify tiles into full, cut, and irregular types
Small Cut Detection: Identify problematic small cut tiles
Wastage Analysis: Calculate material wastage and optimization metrics
Comprehensive Reporting: Export detailed reports with visualizations
Installation
Prerequisites
Python 3.9 or higher
pip (Python package installer)
Setup
Clone the repository or download the source code:
bash
git clone https://github.com/yourusername/tile-layout-app.git
cd tile-layout-app
Create a virtual environment (recommended):
bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install dependencies:
bash
pip install -r requirements.txt
Run the application:
bash
python app.py
Open your web browser and navigate to:
http://127.0.0.1:5000/
Usage Guide
The application follows a step-by-step workflow:

Step 1: Load DXF File
Upload a DXF file containing room boundaries and start points
The application processes the file and identifies rooms and start points
Rooms are automatically clustered into apartments
Default names are assigned
Step 2: Name Rooms
Assign meaningful names to apartments and rooms
These names will be used throughout the process and in reports
Step 3: Set Apartment Orientation
Set orientation for each apartment (0° or 90°)
This determines how tiles will be aligned
Step 4: Tile Coverage
Configure tile layout settings:
Grout thickness
Whether SP layer dimensions include grout
Layout type (standard or staggered)
Stagger percentage and direction
Generate tile layout with proper coverage
Step 5: Tile Classification
Classify tiles based on their type:
Full tiles
Cut tiles (X-direction, Y-direction, or combined)
Irregular tiles
Choose between pattern-based or flat tile classification
Step 6: Identify Small Cut Tiles
Set size threshold to identify problematic small cut tiles
Choose whether to exclude small cuts from future processing
Visualize and analyze the distribution of small cuts
Step 7: Export with Wastage Analysis
Generate a comprehensive final report
Includes material wastage analysis, area calculations, tile inventory
Available in Excel or CSV format
Detailed breakdowns for planning and ordering
File Structure
tile-layout-app/
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       ├── main.js
│       ├── dxfProcessor.js
│       ├── roomProcessor.js
│       ├── tileProcessor.js
│       ├── visualization.js
│       └── export.js
├── templates/
│   ├── index.html
│   ├── step1.html
│   ├── step2.html
│   ├── step3.html
│   ├── step4.html
│   ├── step5.html
│   ├── step6.html
│   └── step7.html
├── uploads/            # Uploaded DXF files are stored here
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
└── README.md           # This file
DXF File Requirements
For optimal results, your DXF file should include:

Room boundaries on a layer named "Tile Layout" or similar
Starting points for tile placement on a layer named "SP"
Start points should be closed polylines to indicate tile size
Clear separation between different rooms
Customization
The application is designed to be customizable:

Modify the CSS to change the appearance
Adjust the step-by-step workflow as needed
Extend functionality by adding new modules
Add support for additional DXF features
Troubleshooting
Common Issues
DXF parsing errors:
Ensure your DXF file is valid and contains the necessary layers
Check that room boundaries are closed polylines
Tile coverage issues:
Verify that start points are correctly positioned within rooms
Check grout thickness settings
Application crashes:
Check Python version compatibility
Verify all dependencies are installed correctly
Debug Mode
To run the application in debug mode with detailed logs:

bash
python app.py --debug
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Based on Python code for DXF processing and tile layout analysis
Built with Flask, NumPy, Pandas, Shapely, and ezdxf
Visualization created with Matplotlib
Contact
For questions or support, please reach out to your.email@example.com or open an issue on the project repository.

