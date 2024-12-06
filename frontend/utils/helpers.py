import requests
import tempfile
import os
from config import API_URL

def process_upload(uploaded_file):
    """Process the uploaded file and send to backend for analysis"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Send file to backend
        with open(tmp_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/analyze", files=files)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Log the response status and content
        print(f"Response Status: {response.status_code}")
        print(f"Response Content: {response.text}")
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except Exception as e:
        print(f"Error processing upload: {str(e)}")
        return None