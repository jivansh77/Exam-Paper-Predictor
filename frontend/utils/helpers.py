import requests
import tempfile
import os
from config import API_URL

def process_upload(uploaded_files):
    """Process the uploaded files and send to backend for analysis"""
    try:
        files = []
        temp_files = []
        
        # Create temporary files
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_files.append(tmp_file.name)
                files.append(('files[]', open(tmp_file.name, 'rb')))
        
        # Send files to backend
        response = requests.post(f"{API_URL}/analyze", files=files)
        
        # Clean up temp files
        for temp_file in temp_files:
            os.unlink(temp_file)
            
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error processing upload: {str(e)}")
        return None
    finally:
        # Ensure all file handles are closed
        for _, f in files:
            f.close()