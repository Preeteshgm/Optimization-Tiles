FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (including those for Shapely and Matplotlib)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgeos-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads logs exports

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 5000

# Run the application
CMD ["python", "run_server.py"]
