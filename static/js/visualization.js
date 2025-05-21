// Visualization JavaScript file
// Handles visualization of rooms, tiles, and analysis results

class VisualizationProcessor {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.isDragging = false;
        this.lastX = 0;
        this.lastY = 0;
    }

    /**
     * Initialize the visualization canvas
     * @param {HTMLElement} canvasContainer - Container for the canvas
     * @param {Number} width - Canvas width
     * @param {Number} height - Canvas height
     */
    initCanvas(canvasContainer, width = 800, height = 600) {
        if (!canvasContainer) return;
        
        // Create canvas element
        this.canvas = document.createElement('canvas');
        this.canvas.width = width;
        this.canvas.height = height;
        this.canvas.style.border = '1px solid #ccc';
        
        // Store original dimensions as data attributes
        this.canvas.setAttribute('data-original-width', width);
        this.canvas.setAttribute('data-original-height', height);
        
        // Get 2D context
        this.ctx = this.canvas.getContext('2d');
        
        // Add canvas to container
        canvasContainer.innerHTML = '';
        canvasContainer.appendChild(this.canvas);
        
        // Add event listeners for pan/zoom
        this.setupPanZoom();
        
        // Initial clear
        this.clear();
    }

    /**
     * Set up pan and zoom functionality
     */
    setupPanZoom() {
        if (!this.canvas) return;
        
        // Mouse wheel for zoom
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Zoom factor (larger for faster zoom)
            const factor = e.deltaY < 0 ? 1.1 : 0.9;
            
            // Apply zoom
            this.ctx.translate(x, y);
            this.ctx.scale(factor, factor);
            this.ctx.translate(-x, -y);
            
            // Update scale
            this.scale *= factor;
            
            // Redraw
            this.redraw();
        });
        
        // Mouse down for pan
        this.canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.lastX = e.offsetX;
            this.lastY = e.offsetY;
            this.canvas.style.cursor = 'grabbing';
        });
        
        // Mouse move for pan
        this.canvas.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            
            const dx = e.offsetX - this.lastX;
            const dy = e.offsetY - this.lastY;
            
            this.ctx.translate(dx, dy);
            
            this.lastX = e.offsetX;
            this.lastY = e.offsetY;
            
            // Update offset
            this.offsetX += dx;
            this.offsetY += dy;
            
            // Redraw
            this.redraw();
        });
        
        // Mouse up to end pan
        this.canvas.addEventListener('mouseup', () => {
            this.isDragging = false;
            this.canvas.style.cursor = 'default';
        });
        
        // Mouse leave to end pan
        this.canvas.addEventListener('mouseleave', () => {
            this.isDragging = false;
            this.canvas.style.cursor = 'default';
        });
        
        // Double click to reset view
        this.canvas.addEventListener('dblclick', () => {
            this.resetView();
        });
    }

    /**
     * Reset view to original state
     */
    resetView() {
        // Reset transformation
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        
        // Reset variables
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        
        // Redraw
        this.redraw();
    }

    /**
     * Clear the canvas
     */
    clear() {
        if (!this.ctx) return;
        
        // Save current transformation
        this.ctx.save();
        
        // Reset transformation to clear entire canvas
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        
        // Clear
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Restore transformation
        this.ctx.restore();
    }

    /**
     * Redraw the canvas with current data
     */
    redraw() {
        // This would be overridden in specific visualizations
        this.clear();
    }

    /**
     * Draw room boundaries
     * @param {Array} rooms - Array of room data
     * @param {Object} options - Drawing options
     */
    drawRooms(rooms, options = {}) {
        if (!this.ctx || !rooms) return;
        
        const defaultOptions = {
            strokeColor: '#000',
            strokeWidth: 2,
            fillColor: 'rgba(200, 200, 255, 0.3)',
            labelColor: '#000',
            labelSize: 14,
            drawLabels: true
        };
        
        const opts = { ...defaultOptions, ...options };
        
        // Clear canvas
        this.clear();
        
        // Draw each room
        rooms.forEach(room => {
            if (!room.polygon) return;
            
            const points = room.polygon.coordinates[0];
            
            // Begin path
            this.ctx.beginPath();
            
            // Move to first point
            this.ctx.moveTo(points[0][0], points[0][1]);
            
            // Draw lines to each point
            for (let i = 1; i < points.length; i++) {
                this.ctx.lineTo(points[i][0], points[i][1]);
            }
            
            // Close path
            this.ctx.closePath();
            
            // Fill
            this.ctx.fillStyle = opts.fillColor;
            this.ctx.fill();
            
            // Stroke
            this.ctx.strokeStyle = opts.strokeColor;
            this.ctx.lineWidth = opts.strokeWidth;
            this.ctx.stroke();
            
            // Draw label if enabled
            if (opts.drawLabels && room.room_name) {
                this.ctx.fillStyle = opts.labelColor;
                this.ctx.font = `${opts.labelSize}px Arial`;
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                
                // Draw at centroid
                this.ctx.fillText(room.room_name, room.centroid_x, room.centroid_y);
            }
        });
    }

    /**
     * Draw tiles
     * @param {Array} tiles - Array of tile data
     * @param {Object} options - Drawing options
     */
    drawTiles(tiles, options = {}) {
        if (!this.ctx || !tiles) return;
        
        const defaultOptions = {
            strokeColor: '#000',
            strokeWidth: 1,
            groutColor: '#fff',
            groutWidth: 3,
            colorMap: {
                'full': 'rgba(0, 200, 0, 0.7)',
                'cut_x': 'rgba(255, 165, 0, 0.7)',
                'cut_y': 'rgba(255, 0, 0, 0.7)',
                'all_cut': 'rgba(128, 0, 128, 0.7)',
                'irregular': 'rgba(0, 0, 255, 0.7)',
                'small': 'rgba(255, 0, 0, 0.9)'
            },
            drawLabels: false,
            labelColor: '#fff',
            labelSize: 10
        };
        
        const opts = { ...defaultOptions, ...options };
        
        // Clear canvas first if this is not an overlay
        if (!options.overlay) {
            this.clear();
        }
        
        // Draw each tile
        tiles.forEach(tile => {
            if (!tile.polygon) return;
            
            const type = tile.classification || tile.type || 'unknown';
            const points = tile.polygon.coordinates[0];
            
            // Begin path
            this.ctx.beginPath();
            
            // Move to first point
            this.ctx.moveTo(points[0][0], points[0][1]);
            
            // Draw lines to each point
            for (let i = 1; i < points.length; i++) {
                this.ctx.lineTo(points[i][0], points[i][1]);
            }
            
            // Close path
            this.ctx.closePath();
            
            // Fill with color based on type
            this.ctx.fillStyle = opts.colorMap[type] || '#ccc';
            this.ctx.fill();
            
            // Stroke
            this.ctx.strokeStyle = opts.strokeColor;
            this.ctx.lineWidth = opts.strokeWidth;
            this.ctx.stroke();
            
            // Draw label if enabled
            if (opts.drawLabels && tile.label) {
                // Get centroid
                let cx = 0, cy = 0;
                for (let i = 0; i < points.length - 1; i++) {
                    cx += points[i][0];
                    cy += points[i][1];
                }
                cx /= (points.length - 1);
                cy /= (points.length - 1);
                
                this.ctx.fillStyle = opts.labelColor;
                this.ctx.font = `${opts.labelSize}px Arial`;
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                
                // Draw at centroid
                this.ctx.fillText(tile.label, cx, cy);
            }
        });
    }

    /**
     * Draw the legend
     * @param {Object} colorMap - Map of types to colors
     * @param {Object} options - Drawing options
     */
    drawLegend(colorMap, options = {}) {
        if (!this.ctx || !colorMap) return;
        
        const defaultOptions = {
            x: 20,
            y: 20,
            boxSize: 20,
            spacing: 30,
            labelColor: '#000',
            labelSize: 14
        };
        
        const opts = { ...defaultOptions, ...options };
        
        // Draw each legend item
        let y = opts.y;
        
        Object.entries(colorMap).forEach(([type, color]) => {
            // Draw color box
            this.ctx.fillStyle = color;
            this.ctx.fillRect(opts.x, y, opts.boxSize, opts.boxSize);
            
            // Draw stroke
            this.ctx.strokeStyle = '#000';
            this.ctx.lineWidth = 1;
            this.ctx.strokeRect(opts.x, y, opts.boxSize, opts.boxSize);
            
            // Draw label
            this.ctx.fillStyle = opts.labelColor;
            this.ctx.font = `${opts.labelSize}px Arial`;
            this.ctx.textAlign = 'left';
            this.ctx.textBaseline = 'middle';
            
            // Capitalize type and format label
            const label = type.charAt(0).toUpperCase() + type.slice(1).replace('_', ' ');
            this.ctx.fillText(label, opts.x + opts.boxSize + 10, y + opts.boxSize / 2);
            
            // Increment y for next item
            y += opts.spacing;
        });
    }

    /**
     * Export canvas to base64 image
     * @param {String} format - Image format (default: 'png')
     * @returns {String} Base64 encoded image data
     */
    exportToBase64(format = 'png') {
        if (!this.canvas) return null;
        
        return this.canvas.toDataURL(`image/${format}`);
    }

    /**
     * Export canvas to blob
     * @param {String} format - Image format (default: 'png')
     * @param {Function} callback - Callback with the blob
     */
    exportToBlob(format = 'png', callback) {
        if (!this.canvas) {
            callback(null);
            return;
        }
        
        this.canvas.toBlob(callback, `image/${format}`);
    }

    /**
     * Create visualization from data URL
     * @param {String} dataUrl - Data URL of the image
     * @param {HTMLElement} container - Container element
     */
    createFromDataUrl(dataUrl, container) {
        if (!container) return;
        
        const img = document.createElement('img');
        img.src = dataUrl;
        img.className = 'img-fluid';
        img.alt = 'Visualization';
        
        container.innerHTML = '';
        container.appendChild(img);
    }
}

// Export the VisualizationProcessor class
window.VisualizationProcessor = VisualizationProcessor;