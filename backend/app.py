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
    logging.info('Received request to analyze paper.')
    
    if 'file' not in request.files:
        logging.error('No file provided')
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    logging.info(f'Received file: {file.filename}')
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logging.info(f'File saved to {filepath}')
        
        try:
            processed_data = processor.process_paper(filepath)
            # Convert numpy arrays to lists in processed_data
            processed_data['clusters'] = processed_data['clusters'].tolist()
            processed_data['embeddings'] = processed_data['embeddings'].tolist()
            
            analysis_results = analyzer.analyze(processed_data)
            logging.info('File processed successfully.')
        except Exception as e:
            logging.error(f'Error processing the file: {e}')
            return jsonify({'error': 'Error processing the file', 'details': str(e)}), 500
        
        os.remove(filepath)
        return jsonify(analysis_results)
    
    logging.error('Invalid file type')
    return jsonify({'error': 'Invalid file type'}), 400

# Rest of the code remains the same
if __name__ == '__main__':
    app.run(debug=True)