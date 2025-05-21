// Tile Processor JavaScript file
// Handles tile generation, classification, and analysis

class TileProcessor {
    constructor() {
        this.tiles = [];
        this.tileLayout = null;
        this.classification = null;
        this.smallCuts = null;
    }

    /**
     * Generate tile layout based on given parameters
     * @param {Object} params - Tile generation parameters
     * @param {Function} onComplete - Callback when generation is complete
     */
    generateTileLayout(params, onComplete) {
        // Show loading overlay
        if (window.showLoading) {
            window.showLoading('Generating tile layout...');
        }

        // Send parameters to server
        fetch('/step4', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading overlay
            if (window.hideLoading) {
                window.hideLoading();
            }

            if (data.error) {
                console.error('Error generating tile layout:', data.error);
                if (onComplete) {
                    onComplete(false, data.error);
                }
            } else {
                // Store tile layout
                this.tileLayout = data.tile_plot;
                
                if (onComplete) {
                    onComplete(true, data);
                }
            }
        })
        .catch(error => {
            // Hide loading overlay
            if (window.hideLoading) {
                window.hideLoading();
            }

            console.error('Error generating tile layout:', error);
            if (onComplete) {
                onComplete(false, 'Network error: ' + error.message);
            }
        });
    }

    /**
     * Classify tiles based on given parameters
     * @param {Object} params - Classification parameters
     * @param {Function} onComplete - Callback when classification is complete
     */
    classifyTiles(params, onComplete) {
        // Show loading overlay
        if (window.showLoading) {
            window.showLoading('Classifying tiles...');
        }

        // Send parameters to server
        fetch('/step5', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading overlay
            if (window.hideLoading) {
                window.hideLoading();
            }

            if (data.error) {
                console.error('Error classifying tiles:', data.error);
                if (onComplete) {
                    onComplete(false, data.error);
                }
            } else {
                // Store classification data
                this.classification = data.classification_plot;
                this.hasPattern = params.has_pattern;
                
                if (onComplete) {
                    onComplete(true, data);
                }
            }
        })
        .catch(error => {
            // Hide loading overlay
            if (window.hideLoading) {
                window.hideLoading();
            }

            console.error('Error classifying tiles:', error);
            if (onComplete) {
                onComplete(false, 'Network error: ' + error.message);
            }
        });
    }

    /**
     * Identify small cut tiles
     * @param {Object} params - Small cuts identification parameters
     * @param {Function} onComplete - Callback when identification is complete
     */
    identifySmallCuts(params, onComplete) {
        // Show loading overlay
        if (window.showLoading) {
            window.showLoading('Identifying small cut tiles...');
        }

        // Send parameters to server
        fetch('/step6', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading overlay
            if (window.hideLoading) {
                window.hideLoading();
            }

            if (data.error) {
                console.error('Error identifying small cuts:', data.error);
                if (onComplete) {
                    onComplete(false, data.error);
                }
            } else {
                // Store small cuts data
                this.smallCuts = data.small_tiles_plot;
                this.sizeThreshold = params.size_threshold;
                this.excludeSmallCuts = params.exclude_small_cuts;
                
                if (onComplete) {
                    onComplete(true, data);
                }
            }
        })
        .catch(error => {
            // Hide loading overlay
            if (window.hideLoading) {
                window.hideLoading();
            }

            console.error('Error identifying small cuts:', error);
            if (onComplete) {
                onComplete(false, 'Network error: ' + error.message);
            }
        });
    }

    /**
     * Export final report with wastage analysis
     * @param {Object} params - Export parameters
     * @param {Function} onComplete - Callback when export is complete
     */
    exportReport(params, onComplete) {
        // Show loading overlay
        if (window.showLoading) {
            window.showLoading('Generating final report...');
        }

        // Send parameters to server
        fetch('/step7', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading overlay
            if (window.hideLoading) {
                window.hideLoading();
            }

            if (data.error) {
                console.error('Error generating report:', data.error);
                if (onComplete) {
                    onComplete(false, data.error);
                }
            } else {
                // Store export results
                this.exportResults = data.results;
                
                if (onComplete) {
                    onComplete(true, data);
                }
            }
        })
        .catch(error => {
            // Hide loading overlay
            if (window.hideLoading) {
                window.hideLoading();
            }

            console.error('Error generating report:', error);
            if (onComplete) {
                onComplete(false, 'Network error: ' + error.message);
            }
        });
    }

    /**
     * Update tile layout display
     * @param {HTMLElement} layoutElement - Element to display tile layout
     */
    updateTileLayoutDisplay(layoutElement) {
        if (layoutElement && this.tileLayout) {
            const img = layoutElement.querySelector('img');
            if (img) {
                img.src = 'data:image/png;base64,' + this.tileLayout;
            }
        }
    }

    /**
     * Update classification display
     * @param {HTMLElement} classificationElement - Element to display classification
     */
    updateClassificationDisplay(classificationElement) {
        if (classificationElement && this.classification) {
            const img = classificationElement.querySelector('img');
            if (img) {
                img.src = 'data:image/png;base64,' + this.classification;
            }
        }
    }

    /**
     * Update small cuts display
     * @param {HTMLElement} smallCutsElement - Element to display small cuts
     */
    updateSmallCutsDisplay(smallCutsElement) {
        if (smallCutsElement && this.smallCuts) {
            const img = smallCutsElement.querySelector('img');
            if (img) {
                img.src = 'data:image/png;base64,' + this.smallCuts;
            }
        }
    }

    /**
     * Generate example statistics for classification
     * @returns {Object} Example statistics
     */
    getExampleClassificationStats() {
        // This is example data - would be replaced with real data in production
        const stats = [
            { type: 'Full', count: 180, percentage: 72 },
            { type: 'Irregular', count: 15, percentage: 6 }
        ];
        
        if (this.hasPattern) {
            stats.push({ type: 'Cut X', count: 30, percentage: 12 });
            stats.push({ type: 'Cut Y', count: 25, percentage: 10 });
        } else {
            stats.push({ type: 'All Cut', count: 55, percentage: 22 });
        }
        
        return stats;
    }

    /**
     * Generate example data for small cuts
     * @returns {Object} Example small cuts data
     */
    getExampleSmallCutsData() {
        return {
            count: 10,
            percentage: 18.2,
            threshold: this.sizeThreshold || 10,
            excluded: this.excludeSmallCuts || true,
            distribution: {
                byRoom: [
                    {apartment: 'A1', room: 'Bathroom', count: 4},
                    {apartment: 'A1', room: 'Kitchen', count: 2},
                    {apartment: 'A2', room: 'Bathroom', count: 3},
                    {apartment: 'A3', room: 'Kitchen', count: 1}
                ],
                bySize: [
                    {range: '0-2mm', count: 1},
                    {range: '2-4mm', count: 2},
                    {range: '4-6mm', count: 3},
                    {range: '6-8mm', count: 2},
                    {range: '8-10mm', count: 2}
                ]
            },
            list: [
                {apartment: 'A1', room: 'Bathroom', type: 'Cut X', dimension: 2.5},
                {apartment: 'A1', room: 'Bathroom', type: 'Cut Y', dimension: 4.7},
                {apartment: 'A1', room: 'Bathroom', type: 'Cut X', dimension: 5.2},
                {apartment: 'A1', room: 'Bathroom', type: 'Cut Y', dimension: 8.3},
                {apartment: 'A1', room: 'Kitchen', type: 'Cut X', dimension: 3.1},
                {apartment: 'A1', room: 'Kitchen', type: 'Cut Y', dimension: 9.8},
                {apartment: 'A2', room: 'Bathroom', type: 'Cut X', dimension: 1.5},
                {apartment: 'A2', room: 'Bathroom', type: 'Cut Y', dimension: 5.5},
                {apartment: 'A2', room: 'Bathroom', type: 'Cut X', dimension: 7.2},
                {apartment: 'A3', room: 'Kitchen', type: 'Cut X', dimension: 6.9}
            ]
        };
    }

    /**
     * Generate example data for export results
     * @returns {Object} Example export results
     */
    getExampleExportResults() {
        return {
            file_name: this.exportResults?.file_name || "final_tiles_export_FINAL_REPORT.xlsx",
            wastage: [
                {apartment: 'A1', area: 15.75, percentage: 4.8},
                {apartment: 'A2', area: 12.30, percentage: 5.2},
                {apartment: 'A3', area: 10.45, percentage: 6.5},
                {apartment: 'Total', area: 38.50, percentage: 5.2}
            ],
            summary: [
                {type: 'Full Tiles', count: 180},
                {type: 'Irregular Tiles', count: 10},
                {type: 'Cut Tiles (X)', count: 30},
                {type: 'Cut Tiles (Y)', count: 25},
                {type: 'Small Cut Tiles (excluded)', count: 10},
                {type: 'Total Tiles', count: 255}
            ]
        };
    }
}

// Export the TileProcessor class
window.TileProcessor = TileProcessor;