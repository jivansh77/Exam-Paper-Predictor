import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv('API_URL', 'http://127.0.0.1:5000')

# Add any other configuration variables here
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size