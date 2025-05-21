// Export Processor JavaScript file
// Handles export of tile data and analysis results

class ExportProcessor {
    constructor() {
        this.data = null;
        this.exportResults = null;
    }

    /**
     * Initialize with data
     * @param {Object} data - Data to export
     */
    init(data) {
        this.data = data;
    }

    /**
     * Export data with wastage analysis
     * @param {Object} params - Export parameters
     * @param {Function} onComplete - Callback when export is complete
     */
    exportWithWastageAnalysis(params, onComplete) {
        // Show loading overlay
        if (window.showLoading) {
            window.showLoading('Generating export with wastage analysis...');
        }

        // Send data to server
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
                console.error('Error exporting data:', data.error);
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

            console.error('Error exporting data:', error);
            if (onComplete) {
                onComplete(false, 'Network error: ' + error.message);
            }
        });
    }

    /**
     * Get export results
     * @returns {Object} Export results
     */
    getExportResults() {
        return this.exportResults;
    }

    /**
     * Update export results display
     * @param {HTMLElement} reportElement - Element to display report name
     * @param {HTMLElement} downloadButton - Download button element
     * @param {HTMLElement} wastageTable - Wastage table element
     * @param {HTMLElement} summaryTable - Summary table element
     */
    updateDisplay(reportElement, downloadButton, wastageTable, summaryTable) {
        if (!this.exportResults) return;
        
        // Update report filename
        if (reportElement) {
            reportElement.textContent = this.exportResults.file_name;
        }
        
        // Update download button
        if (downloadButton) {
            downloadButton.href = `/download/${this.exportResults.file_name}`;
        }
        
        // Update wastage table with example data (would be real data in production)
        if (wastageTable) {
            wastageTable.innerHTML = `
                <tr><td>A1</td><td>15.75</td><td>4.8%</td></tr>
                <tr><td>A2</td><td>12.30</td><td>5.2%</td></tr>
                <tr><td>A3</td><td>10.45</td><td>6.5%</td></tr>
                <tr><td><strong>Total</strong></td><td><strong>38.50</strong></td><td><strong>5.2%</strong></td></tr>
            `;
        }
        
        // Update summary table
        if (summaryTable) {
            summaryTable.innerHTML = `
                <tr><td>Full Tiles</td><td>180</td></tr>
                <tr><td>Irregular Tiles</td><td>10</td></tr>
                <tr><td>Cut Tiles (X)</td><td>30</td></tr>
                <tr><td>Cut Tiles (Y)</td><td>25</td></tr>
                <tr><td>Small Cut Tiles (excluded)</td><td>10</td></tr>
                <tr><td><strong>Total Tiles</strong></td><td><strong>255</strong></td></tr>
            `;
        }
    }

    /**
     * Generate CSV data
     * @param {Array} data - Array of objects
     * @param {Array} headers - Array of header names
     * @returns {String} CSV data
     */
    generateCSV(data, headers) {
        if (!data || !headers) return '';
        
        // Create header row
        let csv = headers.join(',') + '\n';
        
        // Add data rows
        data.forEach(row => {
            const values = headers.map(header => {
                const value = row[header] || '';
                // Quote strings with commas
                return typeof value === 'string' && value.includes(',') ? 
                    `"${value}"` : value;
            });
            csv += values.join(',') + '\n';
        });
        
        return csv;
    }

    /**
     * Download data as CSV
     * @param {Array} data - Array of objects
     * @param {Array} headers - Array of header names
     * @param {String} filename - File name
     */
    downloadCSV(data, headers, filename) {
        const csv = this.generateCSV(data, headers);
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.style.display = 'none';
        
        document.body.appendChild(a);
        a.click();
        
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 100);
    }

    /**
     * Create download link for CSV data
     * @param {Array} data - Array of objects
     * @param {Array} headers - Array of header names
     * @param {String} filename - File name
     * @returns {String} Download link URL
     */
    createCSVDownloadLink(data, headers, filename) {
        const csv = this.generateCSV(data, headers);
        const blob = new Blob([csv], { type: 'text/csv' });
        return URL.createObjectURL(blob);
    }

    /**
     * Estimate file size
     * @param {Object} data - Export data
     * @param {String} format - Export format ('excel' or 'csv')
     * @returns {String} Human-readable file size
     */
    estimateFileSize(data, format) {
        if (!data) return '0 KB';
        
        // Estimate size based on data
        let sizeBytes;
        
        if (format === 'excel') {
            // Excel files have overhead
            sizeBytes = JSON.stringify(data).length * 1.5;
        } else {
            // CSV is more efficient
            sizeBytes = JSON.stringify(data).length;
        }
        
        // Convert to human-readable
        if (sizeBytes < 1024) {
            return `${sizeBytes} B`;
        } else if (sizeBytes < 1024 * 1024) {
            return `${(sizeBytes / 1024).toFixed(1)} KB`;
        } else {
            return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
        }
    }
}

// Export the ExportProcessor class
window.ExportProcessor = ExportProcessor;