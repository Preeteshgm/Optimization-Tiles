// Room Processor JavaScript file
// Handles room clustering, apartment naming, and orientation

class RoomProcessor {
    constructor() {
        this.rooms = [];
        this.apartments = [];
        this.orientations = {};
    }

    /**
     * Initialize room data from server response
     * @param {Object} roomData - Room data from server
     */
    initFromServerData(roomData) {
        if (!roomData) return;
        
        this.rooms = roomData;
        this.processApartments();
    }

    /**
     * Process rooms into apartments
     */
    processApartments() {
        // Group rooms by apartment name
        const apartmentMap = {};
        
        this.rooms.forEach(room => {
            const aptName = room.apartment_name;
            if (!apartmentMap[aptName]) {
                apartmentMap[aptName] = [];
            }
            apartmentMap[aptName].push(room);
        });
        
        // Convert to array format
        this.apartments = Object.keys(apartmentMap).map(aptName => {
            return {
                apartment_name: aptName,
                rooms: apartmentMap[aptName]
            };
        });
    }

    /**
     * Update apartment and room names
     * @param {Array} updatedApartments - Array of updated apartment data
     * @param {Function} onComplete - Callback when update is complete
     */
    updateNames(updatedApartments, onComplete) {
        // Show loading overlay
        if (window.showLoading) {
            window.showLoading('Updating names...');
        }
        
        // Send updated names to server
        fetch('/step2', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ apartments: updatedApartments })
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading overlay
            if (window.hideLoading) {
                window.hideLoading();
            }

            if (data.error) {
                console.error('Error updating names:', data.error);
                if (onComplete) {
                    onComplete(false, data.error);
                }
            } else {
                // Update local data
                this.updatedPlot = data.updated_plot;
                
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

            console.error('Error sending name updates:', error);
            if (onComplete) {
                onComplete(false, 'Network error: ' + error.message);
            }
        });
    }

    /**
     * Update apartment orientations
     * @param {Array} orientations - Array of apartment orientations
     * @param {Function} onComplete - Callback when update is complete
     */
    updateOrientations(orientations, onComplete) {
        // Show loading overlay
        if (window.showLoading) {
            window.showLoading('Updating orientations...');
        }
        
        // Send orientations to server
        fetch('/step3', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ orientations: orientations })
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading overlay
            if (window.hideLoading) {
                window.hideLoading();
            }

            if (data.error) {
                console.error('Error updating orientations:', data.error);
                if (onComplete) {
                    onComplete(false, data.error);
                }
            } else {
                // Store orientations locally
                orientations.forEach(orient => {
                    this.orientations[orient.apartment_name] = orient.orientation;
                });
                
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

            console.error('Error sending orientations:', error);
            if (onComplete) {
                onComplete(false, 'Network error: ' + error.message);
            }
        });
    }

    /**
     * Get apartments data
     * @returns {Array} Array of apartments
     */
    getApartments() {
        return this.apartments;
    }

    /**
     * Get orientations data
     * @returns {Object} Orientations by apartment name
     */
    getOrientations() {
        return this.orientations;
    }

    /**
     * Update display with updated plot
     * @param {HTMLElement} plotElement - Element to display updated plot
     */
    updateDisplay(plotElement) {
        if (plotElement && this.updatedPlot) {
            const img = plotElement.querySelector('img');
            if (img) {
                img.src = 'data:image/png;base64,' + this.updatedPlot;
            }
        }
    }
}

// Export the RoomProcessor class
window.RoomProcessor = RoomProcessor;