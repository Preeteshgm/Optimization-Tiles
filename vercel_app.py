import os

# Set environment variable to identify Vercel
os.environ['VERCEL_ENV'] = 'production'

# Import the Flask app from app.py
from app import app

# This is the entry point for Vercel
if __name__ == '__main__':
    app.run()
