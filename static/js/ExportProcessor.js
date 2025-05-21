class ExportProcessor {
    constructor() {
        this.exportFormats = ['excel', 'pdf', 'zip'];
    }

    /**
     * Generate export reports with selected parameters
     * @param {Object} params - Export parameters
     * @param {Function} callback - Callback function for results
     */
    generateReports(params, callback) {
        // Send request to server
        fetch('/step9/generate_reports', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        })
        .then(response => response.json())
        .then(data => {
            callback(true, data);
        })
        .catch(error => {
            console.error('Error generating reports:', error);
            callback(false, { error: 'Error generating reports: ' + error.message });
        });
    }

    /**
     * Download a specific report
     * @param {string} reportType - Type of report to download
     * @param {string} apartmentId - Optional apartment ID for apartment-specific reports
     */
    downloadReport(reportType, apartmentId = null) {
        let url = `/step9/download_report?type=${reportType}`;
        if (apartmentId) {
            url += `&apartment=${apartmentId}`;
        }
        window.location.href = url;
    }

    /**
     * Download all reports as a package
     */
    downloadAllReports() {
        window.location.href = '/step9/download_all_reports';
    }

    /**
     * Preview a specific report
     * @param {string} reportType - Type of report to preview
     * @param {string} apartmentId - Optional apartment ID for apartment-specific reports
     * @param {Function} callback - Callback function for results
     */
    previewReport(reportType, apartmentId = null, callback) {
        let url = `/step9/preview_report?type=${reportType}`;
        if (apartmentId) {
            url += `&apartment=${apartmentId}`;
        }
        
        fetch(url)
            .then(response => response.json())
            .then(data => {
                callback(true, data);
            })
            .catch(error => {
                console.error('Error previewing report:', error);
                callback(false, { error: 'Error previewing report: ' + error.message });
            });
    }
}