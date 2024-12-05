from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from models.processor import ExamPaperProcessor
from models.analyzer import QuestionAnalyzer

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

processor = ExamPaperProcessor()
analyzer = QuestionAnalyzer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/analyze', methods=['POST'])
def analyze_paper():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process PDF
        processed_data = processor.process_paper(filepath)
        
        # Analyze questions
        analysis_results = analyzer.analyze(processed_data)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(analysis_results)
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/topics', methods=['GET'])
def get_topics():
    topics = analyzer.get_topics()
    return jsonify(topics)

@app.route('/questions/<topic_id>', methods=['GET'])
def get_questions(topic_id):
    questions = analyzer.get_questions_by_topic(topic_id)
    return jsonify(questions)

if __name__ == '__main__':
    app.run(debug=True)