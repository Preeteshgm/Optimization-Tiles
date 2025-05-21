// Optimization Processor JavaScript file
// Handles optimization of cut piece usage with inventory matching

class OptimizationProcessor {
    constructor() {
        this.data = null;
        this.optimizationResults = null;
        this.inventoryData = null;
        this.optimizationParams = null;
        
        console.log('OptimizationProcessor initialized');
        
        // Store intermediate results from each step
        this.cut_data_for_matching = null;
        this.cut_pieces_by_half = null;
        this.match_results = null;
    }

    /**
     * Initialize with data
     * @param {Object} data - Export data from Step 7
     */
    init(data) {
        this.data = data;
    }

    /**
     * Upload and process inventory file
     * @param {File} file - Inventory file to upload
     * @param {Function} onComplete - Callback when upload is complete
     */
    uploadInventory(file, onComplete) {
        if (!file) {
            if (onComplete) onComplete(false, 'No file selected');
            return;
        }

        console.log('Uploading inventory file:', file.name);
        
        // Create form data
        const formData = new FormData();
        formData.append('inventory_file', file);
        
        // Send to server
        fetch('/step8/upload_inventory', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error uploading inventory:', data.error);
                if (onComplete) onComplete(false, data.error);
            } else {
                console.log('Inventory upload successful:', data);
                this.inventoryData = data;
                if (onComplete) onComplete(true, data);
            }
        })
        .catch(error => {
            console.error('Error uploading inventory:', error);
            if (onComplete) onComplete(false, 'Network error: ' + error.message);
        });
    }

    /**
     * Create an inventory template file
     * @param {Function} onComplete - Callback when template creation is complete
     */
    createInventoryTemplate(onComplete) {
        console.log('Creating inventory template...');
        
        // Request template from server
        fetch('/create_inventory_template_route')
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to create template');
            }
            return response.blob();
        })
        .then(blob => {
            // Create download link
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'inventory_template.xlsx';
            document.body.appendChild(a);
            a.click();
            
            // Clean up
            setTimeout(() => {
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }, 100);
            
            if (onComplete) onComplete(true);
        })
        .catch(error => {
            console.error('Error creating template:', error);
            if (onComplete) onComplete(false, error.message);
        });
    }

    /**
     * Run Step 8C: Match cut pieces with progressive tolerance
     * @param {Function} onStepProgress - Progress callback
     * @param {Function} onComplete - Completion callback
     */
    runStep8C(onStepProgress, onComplete) {
        console.log("Running Step 8C: Match cut pieces with progressive tolerance");
        
        // Report initial progress
        if (onStepProgress) onStepProgress('8c', 'running', 10, 'Matching cut pieces with progressive tolerance...');
        
        // Define tolerance ranges as per Colab implementation
        const minTolerance = parseFloat(this.optimizationParams.min_tolerance || 2.0);
        const maxTolerance = parseFloat(this.optimizationParams.max_tolerance || 80.0);
        
        // Create tolerance ranges array - use the same ranges as the Colab implementation
        const toleranceRanges = [10, 20, 40, 60, 80, 100];
        
        // Add more ranges if maxTolerance is higher
        let currentMax = Math.max(...toleranceRanges);
        while (currentMax < maxTolerance) {
            currentMax += 20;  // Increment by 20mm
            toleranceRanges.push(currentMax);
        }
        
        console.log("Using tolerance ranges:", toleranceRanges);
        
        // Get other parameters
        const useInventory = this.optimizationParams.use_inventory || false;
        console.log("Use inventory option:", useInventory);
        
        // Ensure that if useInventory is true, we show that in the log
        if (useInventory) {
            console.log("Inventory matching is ENABLED");
        } else {
            console.log("Inventory matching is DISABLED");
        }
        
        // Call API endpoint for step 8C
        fetch('/step8/match_cut_pieces', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                tolerance_ranges: toleranceRanges,
                min_tolerance: minTolerance,
                max_tolerance: maxTolerance,
                use_inventory: useInventory  // Make sure this is correctly set
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error in Step 8C:', data.error);
                if (onStepProgress) onStepProgress('8c', 'error', 100, `Error: ${data.error}`);
                // Continue to next step anyway
                if (onComplete) onComplete();
            } else {
                console.log('Step 8C successful:', data);
                
                // Store results
                if (data.results) {
                    this.match_results = data.results;
                    this.optimizationResults = data.results; // Also store as optimizationResults for consistency
                    
                    // Show detailed inventory match info
                    if (data.results.inventory_matches) {
                        console.log(`Found ${data.results.inventory_matches} inventory matches`);
                    } else {
                        console.log("No inventory matches found");
                    }
                    
                    // Report progress with match statistics
                    const matchStats = `Found ${data.results.matched_count || 0} total matches: 
                        - Within-apartment: ${data.results.within_apartment_matches || 0}
                        - Cross-apartment: ${data.results.cross_apartment_matches || 0}
                        - Inventory: ${data.results.inventory_matches || 0}
                        - Material saved: ${data.results.total_savings || 0} mm²`;
                    
                    if (onStepProgress) onStepProgress('8c', 'success', 100, matchStats);
                } else {
                    if (onStepProgress) onStepProgress('8c', 'success', 100, 'Cut pieces matched successfully');
                }
                
                // Continue to next step
                if (onComplete) onComplete();
            }
        })
        .catch(error => {
            console.error('Error in Step 8C:', error);
            if (onStepProgress) onStepProgress('8c', 'error', 100, `Network error: ${error.message}`);
            // Continue to next step anyway
            if (onComplete) onComplete();
        });
    }

    /**
     * Run Step 8B: Split cut pieces by half
     * @param {Function} onStepProgress - Progress callback
     * @param {Function} onComplete - Completion callback
     */
    runStep8B(onStepProgress, onComplete) {
        console.log("Running Step 8B: Split cut pieces by half");
        
        // Report initial progress
        if (onStepProgress) onStepProgress('8b', 'running', 10, 'Splitting cut pieces by half threshold...');
        
        // Call API endpoint for step 8B
        fetch('/step8/split_by_half', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                selected_apartments: this.optimizationParams.selected_apartments
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error in Step 8B:', data.error);
                if (onStepProgress) onStepProgress('8b', 'error', 100, `Error: ${data.error}`);
                // Continue to next step anyway
                if (onComplete) onComplete();
            } else {
                console.log('Step 8B successful:', data);
                if (onStepProgress) onStepProgress('8b', 'success', 100, 'Cut pieces split successfully');
                // Store results
                this.cut_pieces_by_half = data.cut_pieces_by_half;
                // Continue to next step
                if (onComplete) onComplete();
            }
        })
        .catch(error => {
            console.error('Error in Step 8B:', error);
            if (onStepProgress) onStepProgress('8b', 'error', 100, `Network error: ${error.message}`);
            // Continue to next step anyway
            if (onComplete) onComplete();
        });
    }

    /**
     * Match cut pieces with progressive tolerance
     * @param {Object} cut_pieces_by_half - The cut pieces divided by half threshold
     * @param {Array} tolerance_ranges - Array of tolerance values to use
     * @param {Object} inventory_data - Optional inventory data
     * @param {Function} onComplete - Callback when matching is complete
     */
    match_cut_pieces(cut_pieces_by_half, tolerance_ranges, inventory_data, onComplete) {
        console.log('Matching cut pieces with progressive tolerance...');
        
        // Create request payload
        const payload = {
            tolerance_ranges: tolerance_ranges,
            use_inventory: inventory_data !== null && inventory_data !== undefined
        };
        
        // Send request to server
        fetch('/step8/match_cut_pieces', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error matching cut pieces:', data.error);
                if (onComplete) onComplete(false, data.error);
            } else {
                console.log('Matching successful:', data);
                this.match_results = data.results;
                if (onComplete) onComplete(true, data.results);
            }
        })
        .catch(error => {
            console.error('Error matching cut pieces:', error);
            if (onComplete) onComplete(false, 'Network error: ' + error.message);
        });
    }

    /**
     * Run Step 8D: Generate match report
     * @param {Function} onStepProgress - Progress callback
     * @param {Function} onComplete - Completion callback
     */
    runStep8D(onStepProgress, onComplete) {
        console.log("Running Step 8D: Generate match report");
        
        // Report progress
        if (onStepProgress) onStepProgress('8d', 'running', 30, 'Generating matching report...');
        
        // In a web app, we can use a simpler version of the report generation
        setTimeout(() => {
            if (this.optimizationResults && this.optimizationResults.matched_count) {
                const totalMatches = this.optimizationResults.matched_count;
                const sameAptMatches = this.optimizationResults.within_apartment_matches;
                const diffAptMatches = this.optimizationResults.cross_apartment_matches;
                const invMatches = this.optimizationResults.inventory_matches;
                
                const reportMsg = `Report generated successfully. Found ${totalMatches} total matches:
    - Same Apartment: ${sameAptMatches}
    - Cross Apartment: ${diffAptMatches}
    - Inventory: ${invMatches}`;
                
                if (onStepProgress) onStepProgress('8d', 'success', 100, reportMsg);
            } else {
                if (onStepProgress) onStepProgress('8d', 'success', 100, 'Report generation complete');
            }
            
            // Continue to next step
            if (onComplete) onComplete();
        }, 500);
    }

    /**
     * Run Step 8E: Export results
     * @param {Function} onStepProgress - Progress callback
     * @param {Function} onComplete - Completion callback
     */
    runStep8E(onStepProgress, onComplete) {
        console.log("Running Step 8E: Export results");
        
        // Report progress
        if (onStepProgress) onStepProgress('8e', 'running', 50, 'Preparing export files...');
        
        // In a web app, we don't need to create the export immediately
        setTimeout(() => {
            // Simulate export preparation
            if (onStepProgress) onStepProgress('8e', 'success', 100, 'Export ready. Use the Download buttons below.');
            
            // Complete the optimization process
            if (onComplete) onComplete();
        }, 500);
    }

    /**
     * Run the complete optimization process
     * @param {Object} params - Optimization parameters 
     * @param {Function} onComplete - Callback when optimization is complete
     */
    runFullOptimization(params, onComplete) {
        console.log('Running full optimization with parameters:', params);
        
        if (!params.selected_apartments || params.selected_apartments.length < 2) {
            if (onComplete) onComplete(false, 'Please select at least two apartments for optimization');
            return;
        }
        
        // Create tolerance ranges
        const minTolerance = parseFloat(params.min_tolerance || 10.0);
        const maxTolerance = parseFloat(params.max_tolerance || 150.0);
        
        // Explicitly log inventory usage
        console.log(`Use inventory checkbox value: ${params.use_inventory}`);
        
        // Include parameters in request
        const requestData = {
            selected_apartments: params.selected_apartments,
            min_tolerance: minTolerance,
            max_tolerance: maxTolerance,
            use_inventory: params.use_inventory === true, // Ensure it's a boolean
            prioritize_same_apartment: params.prioritize_same_apartment || true
        };
        
        console.log('Sending optimization request with data:', requestData);
        
        // Send request to server
        fetch('/step8/full_process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error in optimization:', data.error);
                if (onComplete) onComplete(false, data.error);
            } else {
                console.log('Optimization successful:', data.results);
                this.optimizationResults = data.results;
                
                // Update UI with optimization results
                this.updateUIWithResults(data.results);
                
                if (onComplete) onComplete(true, data.results);
            }
        })
        .catch(error => {
            console.error('Error in optimization:', error);
            if (onComplete) onComplete(false, 'Network error: ' + error.message);
        });
    }

    /**
     * Update UI elements with optimization results
     * @param {Object} results - The optimization results
     */
    updateUIWithResults(results) {
        if (!results) {
            console.warn('No optimization results to display');
            return;
        }
        
        console.log('Updating UI with optimization results');
        
        // Update summary numbers
        const matchedCount = document.getElementById('matchedCount');
        const materialSaved = document.getElementById('materialSaved');
        const withinAptMatches = document.getElementById('withinAptMatches');
        const crossAptMatches = document.getElementById('crossAptMatches');
        const inventoryMatches = document.getElementById('inventoryMatches');
        
        if (matchedCount) matchedCount.textContent = results.matched_count || 0;
        if (materialSaved) materialSaved.textContent = this.formatNumber(results.total_savings || 0);
        if (withinAptMatches) withinAptMatches.textContent = results.within_apartment_matches || 0;
        if (crossAptMatches) crossAptMatches.textContent = results.cross_apartment_matches || 0;
        if (inventoryMatches) inventoryMatches.textContent = results.inventory_matches || 0;
        
        // Update visualization if available
        const optimizationPlot = document.getElementById('optimizationPlot');
        if (optimizationPlot && results.optimization_plot) {
            optimizationPlot.innerHTML = `<img src="data:image/png;base64,${results.optimization_plot}" class="img-fluid" alt="Optimization Visualization">`;
        }
        
        // Process match tables based on pattern mode
        const hasPattern = results.has_pattern;
        
        // Update match tables
        if (hasPattern) {
            // Pattern mode - X and Y direction
            if (results.x_matches && results.x_matches.length > 0) {
                this.updateMatchTable('xMatchesTableBody', results.x_matches);
            }
            
            if (results.y_matches && results.y_matches.length > 0) {
                this.updateMatchTable('yMatchesTableBody', results.y_matches);
            }
            
            // Update unmatched tables
            if (results.x_unmatched_less) {
                this.updateUnmatchedTable('xUnmatchedTableBody', results.x_unmatched_less, 'Less than Half');
            }
            if (results.x_unmatched_more) {
                this.updateUnmatchedTable('xUnmatchedTableBody', results.x_unmatched_more, 'More than Half');
            }
            if (results.y_unmatched_less) {
                this.updateUnmatchedTable('yUnmatchedTableBody', results.y_unmatched_less, 'Less than Half');
            }
            if (results.y_unmatched_more) {
                this.updateUnmatchedTable('yUnmatchedTableBody', results.y_unmatched_more, 'More than Half');
            }
        } else {
            // No pattern mode - All direction
            if (results.all_matches && results.all_matches.length > 0) {
                this.updateMatchTable('matchesTableBody', results.all_matches);
            }
            
            // Update unmatched tables
            if (results.all_unmatched_less) {
                this.updateUnmatchedTable('unmatchedTableBody', results.all_unmatched_less, 'Less than Half');
            }
            if (results.all_unmatched_more) {
                this.updateUnmatchedTable('unmatchedTableBody', results.all_unmatched_more, 'More than Half');
            }
        }
        
        // Apply filters to match tables
        this.filterMatches();
    }

    /**
     * Run the step-by-step optimization process
     * @param {Object} params - Optimization parameters
     * @param {Function} onComplete - Callback when optimization is complete
     * @param {Function} onStepProgress - Callback for step progress updates
     */
    runStepByStepOptimization(params, onComplete, onStepProgress) {
        // Store parameters for use in each step
        this.optimizationParams = params;
        
        // Start with Step 8A
        if (onStepProgress) onStepProgress('8a', 'running', 10, 'Analyzing cut pieces...');
        
        // Call API endpoint for step 8A
        fetch('/step8/analyze_cut_pieces', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                selected_apartments: this.optimizationParams.selected_apartments
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error in Step 8A:', data.error);
                if (onStepProgress) onStepProgress('8a', 'error', 100, `Error: ${data.error}`);
                // Don't continue to next steps if there's an error
                if (onComplete) onComplete(false, data.error);
                return;
            }
            
            console.log('Step 8A successful:', data);
            if (onStepProgress) onStepProgress('8a', 'success', 100, 'Cut pieces analyzed successfully');
            
            // Store the cut data for matching
            this.cut_data_for_matching = data.cut_data;
            
            // Continue to Step 8B
            this.runStep8B(onStepProgress, () => {
                // Continue to Step 8C
                this.runStep8C(onStepProgress, () => {
                    // Continue to Step 8D
                    this.runStep8D(onStepProgress, () => {
                        // Continue to Step 8E
                        this.runStep8E(onStepProgress, () => {
                            if (onComplete) onComplete(true, this.optimizationResults);
                        });
                    });
                });
            });
        })
        .catch(error => {
            console.error('Error in Step 8A:', error);
            if (onStepProgress) onStepProgress('8a', 'error', 100, `Network error: ${error.message}`);
            if (onComplete) onComplete(false, error.message);
        });
    }

    /**
     * Format number with commas for display
     * @param {number} num - Number to format
     * @returns {string} Formatted number
     */
    formatNumber(num) {
        return num.toString().replace(/(\d)(?=(\d{3})+(?!\d))/g, '$1,');
    }

    /**
     * Generate optimization report
     * @param {Function} onComplete - Callback when report generation is complete
     */
    generateOptimizationReport(onComplete) {
        console.log('Generating optimization report...');
        
        // Request report from server
        fetch('/step8/export_matching_results')
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to generate report');
            }
            return response.blob();
        })
        .then(blob => {
            // Create download link
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'matching_results.xlsx';
            document.body.appendChild(a);
            a.click();
            
            // Clean up
            setTimeout(() => {
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }, 100);
            
            if (onComplete) onComplete(true);
        })
        .catch(error => {
            console.error('Error generating report:', error);
            if (onComplete) onComplete(false, error.message);
        });
    }

    /**
     * Download full optimization report package
     * @param {Function} onComplete - Callback when download is complete
     */
    downloadFullReport(onComplete) {
        console.log('Downloading full report package...');
        
        // Request full report from server
        fetch('/step8/download_full_report')
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to generate full report');
            }
            return response.blob();
        })
        .then(blob => {
            // Create download link
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'optimization_report_package.zip';
            document.body.appendChild(a);
            a.click();
            
            // Clean up
            setTimeout(() => {
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }, 100);
            
            if (onComplete) onComplete(true);
        })
        .catch(error => {
            console.error('Error downloading full report:', error);
            if (onComplete) onComplete(false, error.message);
        });
    }
    
    /**
     * Update match tables with filter functionality
     * @param {string} tableId - ID of the table to update
     * @param {Array} matches - Array of match objects
     */
    updateMatchTable(tableId, matches) {
        const tableBody = document.getElementById(tableId);
        if (!tableBody) {
            console.error(`Table body not found with ID: ${tableId}`);
            return;
        }
        
        // Clear existing content
        tableBody.innerHTML = '';
        
        if (!matches || matches.length === 0) {
            console.log(`No matches to display in table ${tableId}`);
            return;
        }
        
        console.log(`Updating ${matches.length} matches in table ${tableId}`);
        
        // Define colors for different match types
        const SAME_APT_COLORS = [
            '#FFD700', '#FFA500', '#FF6347', '#FF1493', '#9932CC', 
            '#4169E1', '#00BFFF', '#00FA9A', '#ADFF2F', '#FFD700'
        ];
        const DIFF_APT_COLOR = '#A9A9A9';  // Gray for different apartment matches
        const INV_COLOR = '#8FBC8F';       // Green for inventory matches
        
        matches.forEach(match => {
            const row = document.createElement('tr');
            
            // Determine match type and color
            let matchType = match['Match Type'] || '';
            let color = '';
            
            if (matchType === 'Inventory') {
                color = INV_COLOR;
                row.dataset.type = 'inventory';
            } else if (matchType === 'Same Apartment') {
                // Extract number from match_id (X1, Y2, etc.)
                const match_id = match['Match ID'] || '';
                const match_num = parseInt(match_id.replace(/\D/g, '')) || 1;
                color = SAME_APT_COLORS[(match_num - 1) % SAME_APT_COLORS.length];
                row.dataset.type = 'same';
            } else { // Cross Apartment
                color = DIFF_APT_COLOR;
                row.dataset.type = 'cross';
            }
            
            // Create the row content
            row.innerHTML = `
                <td><strong>${match['Match ID'] || ''}</strong></td>
                <td>${match['From'] || ''}</td>
                <td>${match['To'] || ''}</td>
                <td>${match['Size (mm)'] || 0} mm</td>
                <td>${match['Waste (mm)'] || 0} mm</td>
                <td>${matchType}</td>
            `;
            
            // Set row style
            row.style.backgroundColor = color + '40';  // Add transparency
            
            // Add to table
            tableBody.appendChild(row);
        });
    }
    
    /**
     * Update unmatched pieces tables
     * @param {string} tableId - ID of the table to update
     * @param {Array} unmatchedPieces - Array of unmatched piece objects
     * @param {string} pieceType - Type of pieces ("Less than Half" or "More than Half")
     */
    updateUnmatchedTable(tableId, unmatchedPieces, pieceType) {
        const tableBody = document.getElementById(tableId);
        if (!tableBody) {
            console.error(`Table body not found with ID: ${tableId}`);
            return;
        }
        
        if (!unmatchedPieces || unmatchedPieces.length === 0) {
            console.log(`No unmatched pieces to display in table ${tableId}`);
            return;
        }
        
        console.log(`Updating ${unmatchedPieces.length} unmatched pieces in table ${tableId}`);
        
        unmatchedPieces.forEach(piece => {
            const row = document.createElement('tr');
            
            row.innerHTML = `
                <td>${piece['Apartment'] || ''}</td>
                <td>${piece['Location'] || ''}</td>
                <td>${piece['Cut Size (mm)'] || 0} mm</td>
                <td>${pieceType}</td>
                <td>${piece['Count'] || 1}</td>
            `;
            
            tableBody.appendChild(row);
        });
    }

    /**
     * Filter matches in tables based on current filter settings
     * @param {boolean} showSame - Show same-apartment matches
     * @param {boolean} showCross - Show cross-apartment matches 
     * @param {boolean} showInventory - Show inventory matches
     * @param {string} searchTerm - Search term to filter by
     */
    filterMatches(showSame = true, showCross = true, showInventory = true, searchTerm = '') {
        // Get all match rows from both tables
        const filterRow = (row) => {
            const rowType = row.dataset.type;
            const shouldShowType = 
                (rowType === 'same' && showSame) || 
                (rowType === 'cross' && showCross) ||
                (rowType === 'inventory' && showInventory);
            
            const rowText = row.textContent.toLowerCase();
            const matchesSearch = searchTerm === '' || rowText.includes(searchTerm.toLowerCase());
            
            row.style.display = shouldShowType && matchesSearch ? '' : 'none';
        };
        
        // Apply filters to all match tables
        ['xMatchesTableBody', 'yMatchesTableBody', 'matchesTableBody'].forEach(tableId => {
            const table = document.getElementById(tableId);
            if (table) {
                Array.from(table.querySelectorAll('tr')).forEach(filterRow);
            }
        });
    }
    
    /**
     * Generate visualization of matches for the report
     * @param {Object} matchResults - Results from the matching process
     * @param {Object} roomDf - DataFrame with room information
     * @returns {string} Base64 encoded image
     */
    generateMatchVisualization(matchResults, roomDf) {
        // This function would normally be implemented in the backend
        // For front-end, we just request visualization from the server
        console.log('Generating visualization of matches...');
        return matchResults.optimization_plot || '';
    }

    /**
     * Export optimization results to Excel
     * @param {Function} callback - Callback function for results
     */
    exportMatchingResults() {
        window.location.href = '/step8/export_matching_results';
    }

    /**
     * Download a full report package
     * @param {Function} callback - Callback function for results
     */
    downloadFullReport() {
        window.location.href = '/step8/download_full_report';
    }

    /**
     * Generate optimization visualization
     * @param {Object} optimizationResults - Optimization results data
     * @param {Function} callback - Callback function for results
     */
    generateVisualization(optimizationResults, callback) {
        // For visualization, we just use the already generated plot in the results
        if (optimizationResults && optimizationResults.optimization_plot) {
            callback(true, { plot: optimizationResults.optimization_plot });
        } else {
            // If no plot is available, we could generate one server-side
            fetch('/step8/generate_visualization', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(optimizationResults)
            })
            .then(response => response.json())
            .then(data => {
                callback(true, data);
            })
            .catch(error => {
                console.error('Error generating visualization:', error);
                callback(false, { error: 'Error generating visualization' });
            });
        }
    }


}

// Export the OptimizationProcessor class
window.OptimizationProcessor = OptimizationProcessor;