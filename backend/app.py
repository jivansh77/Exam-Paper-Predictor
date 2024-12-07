from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import numpy as np
from models.processor import ExamPaperProcessor
from models.analyzer import QuestionAnalyzer
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder  # Set custom JSON encoder
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

processor = ExamPaperProcessor()
analyzer = QuestionAnalyzer()

# Set up logging
logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/analyze', methods=['POST'])
def analyze_paper():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400
        
    temp_paths = []
    
    try:
        # Save all files temporarily
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                temp_paths.append(filepath)
            else:
                return jsonify({'error': 'Invalid file type'}), 400
        
        if not temp_paths:
            return jsonify({'error': 'No valid files to process'}), 400
            
        # Process all papers together
        processed_data = processor.process_multiple_papers(temp_paths)
        analysis_results = analyzer.analyze(processed_data)
        
        return jsonify(analysis_results)
        
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Clean up temporary files
        for path in temp_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logging.error(f"Error removing temporary file {path}: {str(e)}")

# Rest of the code remains the same
if __name__ == '__main__':
    app.run(debug=True)