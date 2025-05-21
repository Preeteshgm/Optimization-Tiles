from waitress import serve
from app import app
import logging
import os
import secrets  # For generating secure secret keys

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging
logging.basicConfig(
    filename='logs/tile_app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# Update your app configuration for production
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True

# Generate a secure secret key if one doesn't exist
SECRET_KEY_FILE = 'secret_key.txt'
if not os.path.exists(SECRET_KEY_FILE):
    with open(SECRET_KEY_FILE, 'w') as f:
        f.write(secrets.token_hex(32))
    logging.info('Generated new secret key')

# Load the secret key
with open(SECRET_KEY_FILE, 'r') as f:
    app.secret_key = f.read().strip()

# Serve the application
if __name__ == '__main__':
    try:
        logging.info('Starting Tile Layout Application server...')
        # Get host IP for display purposes
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        print(f"Server starting on http://{local_ip}:5000")
        print("To access from other networks, you'll need to configure port forwarding")
        
        # Already correct - listening on all interfaces
        serve(app, host='0.0.0.0', port=5000, threads=8)
    except Exception as e:
        logging.error(f'Error starting server: {e}')