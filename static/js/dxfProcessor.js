// DXF Processor JavaScript file
// This file handles parsing and processing of DXF files in the browser

class DxfProcessor {
    constructor() {
        this.rooms = [];
        this.startPoints = [];
        this.tiles = [];
    }

    /**
     * Initialize the DXF processor with file input
     * @param {HTMLElement} fileInput - The file input element
     * @param {Function} onProcessComplete - Callback when processing is complete
     */
    init(fileInput, onProcessComplete) {
        if (!fileInput) {
            console.error('File input element not found');
            return;
        }

        fileInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (!file) return;

            // Show loading overlay
            if (window.showLoading) {
                window.showLoading('Processing DXF file...');
            }

            // Create a FormData object to send the file
            const formData = new FormData();
            formData.append('dxf_file', file);

            // Send file to backend for processing
            fetch('/step1', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading overlay
                if (window.hideLoading) {
                    window.hideLoading();
                }

                if (data.error) {
                    console.error('Error processing DXF file:', data.error);
                    if (onProcessComplete) {
                        onProcessComplete(false, data.error);
                    }
                } else {
                    // Store processed data
                    this.roomCount = data.room_count;
                    this.startPointCount = data.start_point_count;
                    this.tileSizes = data.tile_sizes;
                    this.roomPlot = data.room_plot;
                    this.clusterPlot = data.cluster_plot;

                    if (onProcessComplete) {
                        onProcessComplete(true, data);
                    }
                }
            })
            .catch(error => {
                // Hide loading overlay
                if (window.hideLoading) {
                    window.hideLoading();
                }

                console.error('Error sending DXF file:', error);
                if (onProcessComplete) {
                    onProcessComplete(false, 'Network error: ' + error.message);
                }
            });
        });
    }

    /**
     * Get a summary of the processed DXF file
     * @returns {Object} Summary data
     */
    getSummary() {
        return {
            roomCount: this.roomCount || 0,
            startPointCount: this.startPointCount || 0,
            tileSizes: this.tileSizes || [],
            roomPlot: this.roomPlot,
            clusterPlot: this.clusterPlot
        };
    }

    /**
     * Format tile sizes for display
     * @returns {String} Formatted tile sizes
     */
    getFormattedTileSizes() {
        if (!this.tileSizes || this.tileSizes.length === 0) {
            return 'None detected';
        }

        return this.tileSizes.map(size => `${size[0]}mm × ${size[1]}mm`).join(', ');
    }

    /**
     * Update the display with DXF processing results
     * @param {HTMLElement} summaryElement - Element to display summary
     * @param {HTMLElement} roomPlotElement - Element to display room plot
     * @param {HTMLElement} clusterPlotElement - Element to display cluster plot
     */
    updateDisplay(summaryElement, roomPlotElement, clusterPlotElement) {
        if (summaryElement) {
            const summary = this.getSummary();
            summaryElement.innerHTML = `
                <p>Rooms Found: ${summary.roomCount}</p>
                <p>Start Points Found: ${summary.startPointCount}</p>
                <p>Tile Sizes: ${this.getFormattedTileSizes()}</p>
            `;
        }

        if (roomPlotElement && this.roomPlot) {
            const img = roomPlotElement.querySelector('img');
            if (img) {
                img.src = 'data:image/png;base64,' + this.roomPlot;
            }
        }

        if (clusterPlotElement && this.clusterPlot) {
            const img = clusterPlotElement.querySelector('img');
            if (img) {
                img.src = 'data:image/png;base64,' + this.clusterPlot;
            }
        }
    }
}

// Export the DxfProcessor class
window.DxfProcessor = DxfProcessor;