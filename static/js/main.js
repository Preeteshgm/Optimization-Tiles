// Main JavaScript file for Tile Layout Application

document.addEventListener('DOMContentLoaded', function() {
    // Check for window.history API support
    if (window.history && window.history.pushState) {
        // Add event listener for back/forward navigation
        window.addEventListener('popstate', function(event) {
            // If there's a state and it contains a step
            if (event.state && event.state.step) {
                // Navigate to the specified step
                window.location.href = event.state.url;
            }
        });
    }
    
    // Initialize Bootstrap tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// Function to show loading overlay
function showLoading(message = 'Processing...') {
    const existingOverlay = document.querySelector('.loading-overlay');
    if (existingOverlay) {
        existingOverlay.remove();
    }
    
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    
    const spinnerContainer = document.createElement('div');
    spinnerContainer.className = 'spinner-container';
    
    const spinner = document.createElement('div');
    spinner.className = 'spinner-border text-primary mb-3';
    spinner.setAttribute('role', 'status');
    
    const spinnerText = document.createElement('span');
    spinnerText.className = 'visually-hidden';
    spinnerText.textContent = 'Loading...';
    
    const messageText = document.createElement('p');
    messageText.className = 'mb-0';
    messageText.textContent = message;
    
    spinner.appendChild(spinnerText);
    spinnerContainer.appendChild(spinner);
    spinnerContainer.appendChild(messageText);
    overlay.appendChild(spinnerContainer);
    
    document.body.appendChild(overlay);
}

// Function to hide loading overlay
function hideLoading() {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) {
        overlay.remove();
    }
}

// Helper function to format numbers with commas
function formatNumber(num) {
    return num.toString().replace(/(\d)(?=(\d{3})+(?!\d))/g, '$1,');
}

// Helper function to format percentages
function formatPercentage(value, digits = 1) {
    return parseFloat(value).toFixed(digits) + '%';
}

// Function to validate file input
function validateFileInput(fileInput, allowedExtensions = ['.dxf']) {
    if (!fileInput.files || fileInput.files.length === 0) {
        return { valid: false, message: 'No file selected' };
    }
    
    const file = fileInput.files[0];
    const fileName = file.name;
    const fileExt = '.' + fileName.split('.').pop().toLowerCase();
    
    if (!allowedExtensions.includes(fileExt)) {
        return { 
            valid: false, 
            message: `Invalid file type. Allowed types: ${allowedExtensions.join(', ')}` 
        };
    }
    
    return { valid: true, file: file };
}

// Function to handle form validation
function validateForm(form, validationRules) {
    let isValid = true;
    let firstInvalidElement = null;
    
    // Check each validation rule
    for (const selector in validationRules) {
        const element = form.querySelector(selector);
        if (!element) continue;
        
        const rule = validationRules[selector];
        const errorElement = document.getElementById(rule.errorId);
        
        // Clear previous error
        if (errorElement) {
            errorElement.textContent = '';
            errorElement.classList.add('d-none');
        }
        
        // Apply validation
        let elementValid = true;
        let errorMessage = '';
        
        if (rule.required && !element.value.trim()) {
            elementValid = false;
            errorMessage = rule.requiredMessage || 'This field is required';
        } else if (rule.validator) {
            const validationResult = rule.validator(element);
            elementValid = validationResult.valid;
            errorMessage = validationResult.message || 'Invalid input';
        }
        
        // If invalid, mark and show error
        if (!elementValid) {
            isValid = false;
            
            if (errorElement) {
                errorElement.textContent = errorMessage;
                errorElement.classList.remove('d-none');
            }
            
            // Mark the element
            element.classList.add('is-invalid');
            
            // Store first invalid element for focus
            if (!firstInvalidElement) {
                firstInvalidElement = element;
            }
        } else {
            element.classList.remove('is-invalid');
        }
    }
    
    // Focus on the first invalid element
    if (firstInvalidElement) {
        firstInvalidElement.focus();
    }
    
    return isValid;
}