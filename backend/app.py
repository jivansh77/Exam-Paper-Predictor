from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import torch
import gc
import numpy as np
from models.processor import ExamPaperProcessor
from models.analyzer import QuestionAnalyzer
import logging
from dataclasses import asdict
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

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
app.json_encoder = NumpyEncoder
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global instances
processor = None
analyzer = None

def init_models():
    """Initialize model instances"""
    global processor, analyzer
    
    try:
        logging.info("Initializing models...")
        processor = ExamPaperProcessor()
        analyzer = QuestionAnalyzer()
        logging.info("Models initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing models: {str(e)}")
        raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/analyze', methods=['POST'])
def analyze_paper():
    global processor, analyzer
    
    try:
        if processor is None or analyzer is None:
            init_models()
            
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files[]')
        temp_paths = []
        
        try:
            # Save files temporarily
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    temp_paths.append(filepath)
                    logging.info(f"Saved temporary file: {filepath}")
                else:
                    return jsonify({'error': 'Invalid file type'}), 400
            
            # Process files
            logging.info("Starting paper analysis...")
            processed_data = processor.process_multiple_papers(temp_paths)
            
            if not processed_data.get('questions'):
                return jsonify({'error': 'No questions could be extracted'}), 400
                
            # Analyze the processed data
            analysis_result = analyzer.analyze_papers(processed_data)
            
            response_data = {
                'questions': [
                    {
                        'text': q['text'],
                        'topic': q['topic'],
                        'type': q['type']
                    }
                    for q in processed_data['questions']
                ],
                'topic_distribution': {
                    topic: count for topic, count in Counter(
                        q['topic'] for q in processed_data['questions']
                    ).items()
                },
                'total_questions': len(processed_data['questions']),
                'unique_topics': len(set(q['topic'] for q in processed_data['questions'])),
                'repeated_questions': [
                    {
                        'question': q['text'],
                        'topic': q['topic'],
                        'type': q['type'],
                        'frequency': processed_data['frequencies'].get(q['text'], 1),
                        'confidence': 1.0
                    }
                    for q in processed_data['questions']
                    if processed_data['frequencies'].get(q['text'], 1) > 1
                ],
                'similar_groups': [
                    {
                        'main_question': processed_data['questions'][main_idx]['text'],
                        'similar_questions': [
                            processed_data['questions'][idx]['text'] 
                            for idx in similar_indices
                        ],
                        'topic': processed_data['questions'][main_idx]['topic']
                    }
                    for main_idx, similar_indices in processed_data['similar_groups'].items()
                ],
                'patterns': [
                    f"{count}% of questions are {qtype} type"
                    for qtype, count in Counter(
                        q['type'] for q in processed_data['questions']
                    ).items()
                ]
            }
            
            return jsonify(response_data)
            
        finally:
            # Clean up temporary files
            for path in temp_paths:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        logging.info(f"Removed temporary file: {path}")
                    except Exception as e:
                        logging.error(f"Error removing temporary file {path}: {str(e)}")
                        
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'models_loaded': processor is not None and analyzer is not None})

if __name__ == '__main__':
    init_models()
    app.run(debug=True, use_reloader=False)  # Disable reloader to prevent duplicate model loading