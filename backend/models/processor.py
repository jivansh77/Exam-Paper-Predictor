#LOT OF UNNECESSARY CODE, IMPORTS AND GARBAGE IN THIS, NEED TO CLEAN IT UP

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import pdfplumber
import logging
from contextlib import contextmanager
import gc
import os
import threading
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pytesseract
from PIL import Image, ImageEnhance
import pdf2image
import io
from PIL import ImageEnhance
from huggingface_hub import InferenceClient
from os import getenv
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variable to handle multiprocessing better
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ExamPaperProcessor:
    _instance = None
    _is_initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExamPaperProcessor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._is_initialized:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            logging.info(f"Device set to use {self.device}")
            self._initialize_models()
            self._lock = threading.Lock()
            ExamPaperProcessor._is_initialized = True
            
            # Define topic categories
            self.topic_categories = [
                "Reading Comprehension",
                "Grammar",
                "Vocabulary",
                "Writing",
                "Listening",
                "Speaking",
                "Literature",
                "Poetry",
                "Essay Writing",
                "Critical Analysis"
            ]

    def _extract_questions(self, processed_text):
        """Extract questions from processed text chunks"""
        try:
            logging.info("Extracting questions from text...")
            questions = []
            
            # Convert processed_text to string if it's a list
            if isinstance(processed_text, list):
                text = ' '.join([str(chunk) for chunk in processed_text])
            else:
                text = str(processed_text)
            
            # Process text in chunks to avoid token limits
            chunk_size = 3000  # Increased for Llama-3.2
            text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            for chunk in text_chunks:
                # Generate questions using Llama-3.2
                prompt = f"""<s>[INST] You are an expert in analyzing text (which could be unstructured) and extracting exam questions only with high precision. Your expertise is in Computer Engineering, and you are a graduate of Mumbai University. Carefully identify all questions from the following text.

                - Extract all questions, whether they are MCQs or contain subparts (e.g., i), ii), etc.). Typically, questions in the text are numbered (e.g., Q1, Q2, Q3) and may include subparts.
                - Preserve the original structure and meaning of the questions. If a question is unclear or incomplete, correct it to ensure it makes sense, but do not add any notes or explanations about the changes.
                - Do not output anything other than the extracted questions. Format each question on a new line, starting with 'QUESTION:'.

                Text to analyze: {chunk}
                [/INST]</s>"""

                # Get response from Hugging Face API
                response = self.hf_client.text_generation(
                    prompt,
                    max_new_tokens=1024,
                    temperature=0.2,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    do_sample=True
                )
                
                # Split by "QUESTION:" marker and filter empty strings
                potential_questions = [
                    q.strip() for q in response.split('QUESTION:') 
                    if q.strip()
                ]
                
                for question in potential_questions:
                    # More robust question validation
                    if len(question) > 10 and any(q_word in question.lower() for q_word in [
                        'what', 'why', 'how', 'when', 'where', 'which', 'describe',
                        'explain', 'calculate', 'find', 'determine', 'solve', 'evaluate',
                        'analyze', 'compare', 'discuss', '?'
                    ]):
                        # Clean up the question text
                        cleaned_question = self._clean_question_text(question)
                        
                        # Get topic and type
                        topic = self._get_topic(cleaned_question)
                        q_type = self._classify_question_type(cleaned_question)
                        
                        questions.append({
                            'text': cleaned_question,
                            'topic': topic,
                            'type': q_type
                        })
                        logging.info(f"Extracted question: {cleaned_question[:100]}...")
            
            logging.info(f"Extracted {len(questions)} questions")
            return questions
            
        except Exception as e:
            logging.error(f"Error extracting questions: {str(e)}")
            raise

    def _clean_question_text(self, text):
        """Clean up extracted question text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove any leading numbers or letters used for enumeration
        text = re.sub(r'^\d+\.\s*', '', text)
        text = re.sub(r'^\([a-zA-Z]\)\s*', '', text)
        
        # Ensure the question ends with proper punctuation
        if not text.endswith('?') and not text.endswith('.'):
            text += '?'
        
        return text

    def _is_question(self, text):
        """Check if text is a question"""
        # Basic question detection
        question_indicators = ['?', 'what', 'why', 'how', 'when', 'where', 'which', 'who']
        text_lower = text.lower()
        
        return any(indicator in text_lower for indicator in question_indicators)

    def _classify_question_type(self, text):
        """Classify if a question is theoretical or numerical"""
        # Keywords that suggest numerical questions
        numerical_indicators = [
            'calculate', 'compute', 'solve', 'evaluate', 'find', 
            'how many', 'how much', 'number', 'value', 'percent'
        ]
        
        text_lower = text.lower()
        is_numerical = any(indicator in text_lower for indicator in numerical_indicators)
        
        return 'Numerical' if is_numerical else 'Theory'

    def _get_topic(self, text):
        """Classify the topic of a question using sentence transformer"""
        try:
            # Get embeddings for the question and topics
            question_embedding = self._model_context['sentence_transformer'].encode(text)
            topic_embeddings = self._model_context['sentence_transformer'].encode(self.topic_categories)
            
            # Calculate similarities
            similarities = cosine_similarity(
                [question_embedding], 
                topic_embeddings
            )[0]
            
            # Get the most similar topic
            max_sim_idx = np.argmax(similarities)
            confidence = similarities[max_sim_idx]
            
            if confidence > 0.3:  # Confidence threshold
                return self.topic_categories[max_sim_idx]
            return "General"  # Default topic if no good match
            
        except Exception as e:
            logging.error(f"Error in topic classification: {str(e)}")
            return "General"

    def _calculate_relevance(self, text):
        """Calculate relevance score (0-100)"""
        try:
            # Basic relevance calculation
            # You might want to implement more sophisticated logic
            words = len(text.split())
            return min(max(words * 5, 0), 100)  # Simple score based on length
        except Exception as e:
            logging.error(f"Error calculating relevance: {str(e)}")
            return 50

    def _combine_results(self, results):
        """Combine results from multiple papers"""
        all_questions = []
        all_topics = set()
        all_similar_groups = {}
        all_frequencies = {}
        
        for result in results:
            # Combine questions and topics
            all_questions.extend(result['questions'])
            all_topics.update(result['topics'])
            
            # Merge similar groups
            all_similar_groups.update(result['similar_groups'])
            
            # Merge frequencies
            for text, freq in result['frequencies'].items():
                all_frequencies[text] = all_frequencies.get(text, 0) + freq
        
        return {
            'questions': all_questions,
            'topics': list(all_topics),
            'similar_groups': all_similar_groups,
            'frequencies': all_frequencies
        }

    def process_multiple_papers(self, pdf_paths):
        results = []
        try:
            for pdf_path in pdf_paths:
                logging.info(f"Processing file: {pdf_path}")
                
                # Extract text
                logging.info("Extracting text from PDF...")
                text = self._extract_text(pdf_path)
                
                # Process text
                logging.info("Processing extracted text...")
                processed_text = self._process_text(text)
                
                # Extract questions
                logging.info("Extracting questions...")
                questions = self._extract_questions(text)
                
                # Get unique topics
                topics = list(set(q['topic'] for q in questions))
                
                # Find similar questions
                similar_groups = self._find_similar_questions(questions)
                
                # Calculate frequencies
                frequencies = self._calculate_question_frequencies(questions)
                
                results.append({
                    'questions': questions,
                    'topics': topics,
                    'similar_groups': similar_groups,
                    'frequencies': frequencies
                })
                
            logging.info("Processing completed successfully")
            return self._combine_results(results)
            
        except Exception as e:
            logging.error(f"Error in processing: {str(e)}", exc_info=True)
            raise

    def _extract_text(self, pdf_path):
        """Extract text from PDF using both pdfplumber and OCR"""
        try:
            text = ""
            use_ocr = False
            
            # First try pdfplumber for text-based PDFs
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text and len(page_text.strip()) > 50:  # Check if meaningful text was extracted
                        text += page_text + "\n"
                    else:
                        use_ocr = True
                        break  # Exit the loop if we need to use OCR
            
            # Only use OCR if pdfplumber failed to extract meaningful text
            if use_ocr:
                text = ""  # Reset text before OCR
                images = pdf2image.convert_from_path(pdf_path, dpi=300)
                for img in images:
                    # Preprocess image for better OCR
                    img = self._preprocess_image(img)
                    # Extract text using pytesseract
                    page_text = pytesseract.image_to_string(img, lang='eng')
                    text += page_text + "\n"

            print(use_ocr)
            print(text)
            return text
        except Exception as e:
            logging.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def _preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        img = image.convert('L')
        # Increase contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)
        # Apply thresholding
        img = img.point(lambda x: 0 if x < 128 else 255, '1')
        return img

    def _process_text(self, text):
        """Process text using spaCy"""
        try:
            # Split text into larger chunks to maintain context
            chunk_size = 10000  # Increased from 5000
            overlap = 1000  # Add overlap to avoid cutting questions
            chunks = []
            
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                # If not the first chunk, try to find a good splitting point
                if i > 0:
                    # Find first newline or period after overlap
                    split_point = max(
                        chunk.find('\n', overlap),
                        chunk.find('. ', overlap)
                    )
                    if split_point != -1:
                        chunk = chunk[split_point + 1:]
                chunks.append(chunk)
            
            processed_chunks = []
            for i, chunk in enumerate(chunks, 1):
                logging.info(f"Processing chunk {i}/{len(chunks)}")
                doc = self._nlp(chunk)
                processed_chunks.append(doc)
                
            return processed_chunks
        except Exception as e:
            logging.error(f"Error processing text: {str(e)}")
            raise

    def _initialize_models(self):
        """Initialize required models"""
        try:
            logging.info("Initializing models...")
            
            # Initialize spaCy model
            self._nlp = spacy.load('en_core_web_sm')
            logging.info("Loaded spaCy model")
            
            # Initialize sentence transformer
            self._model_context = {
                'sentence_transformer': SentenceTransformer('all-MiniLM-L6-v2')
            }
            logging.info("Loaded sentence transformer")
            
            # Initialize Hugging Face client
            hf_token = getenv('HUGGINGFACE_API_KEY')
            if not hf_token:
                raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
            
            self.hf_client = InferenceClient(
                token=hf_token,
                model="meta-llama/Llama-3.2-3B-Instruct"
            )
            logging.info("Initialized Hugging Face client")
            
            logging.info("Models initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing models: {str(e)}")
            raise

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF with proper resource handling"""
        try:
            with self._lock:  # Use lock for thread safety
                # Your existing extraction code here
                # Make sure to use self._model_context when needed
                text = ""
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
                
                # Process text with initialized models
                doc = self._nlp(text)
                # Rest of your processing code...
                
                return {
                    'questions': questions,
                    'topics': topics,
                    'similar_groups': similar_groups,
                    'frequencies': frequencies
                }
        except Exception as e:
            logging.error(f"Error extracting questions: {str(e)}")
            raise

    def _process_papers_internal(self, pdf_paths):
        """Process multiple PDF papers"""
        try:
            all_questions = []
            all_texts = []
            
            # First extract text from all PDFs
            for pdf_path in pdf_paths:
                text = self.extract_text_from_pdf(pdf_path)
                if text:
                    all_texts.append(text)
            
            if not all_texts:
                logging.error("No text could be extracted from PDFs")
                return self._empty_response()
            
            # Process each text
            for text in all_texts:
                questions = self.extract_questions(text)
                if questions:
                    all_questions.extend(questions)
            
            if not all_questions:
                logging.error("No questions could be extracted")
                return self._empty_response()
            
            # Process topics and similarities
            topics = self._classify_topics(all_questions)
            similar_groups = self._find_similar_questions(all_questions)
            frequencies = self._calculate_frequencies(all_questions, similar_groups)
            
            return {
                'questions': all_questions,
                'topics': topics,
                'similar_groups': similar_groups,
                'frequencies': frequencies
            }
            
        except Exception as e:
            logging.error(f"Error in process_multiple_papers: {str(e)}")
            return self._empty_response()

    def _empty_response(self):
        """Return empty response structure"""
        return {
            'questions': [],
            'topics': [],
            'similar_groups': {},
            'frequencies': {}
        }

    def extract_questions(self, text):
        """Extract questions using FLAN-T5"""
        try:
            prompt = "Extract exam questions from the following text. Output one question per line:\n\n" + text
            
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad(), self._model_context():
                outputs = self.model.generate(**inputs, max_length=512)
            
            questions = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Split into individual questions and filter empty lines
            questions = [q.strip() for q in questions.split('\n') if q.strip()]
            
            return questions
            
        except Exception as e:
            logging.error(f"Error extracting questions: {str(e)}")
            return []

    def _classify_topics(self, questions):
        """Classify questions into topics"""
        try:
            candidate_topics = [
                "Mathematics", "Physics", "Chemistry", "Biology",
                "Computer Science", "Economics", "History", "Literature",
                "Theory", "Practical Application", "Problem Solving",
                "Conceptual Understanding"
            ]
            
            topics = []
            for question in questions:
                result = self.classifier(question, candidate_topics)
                topics.append({
                    'topic': result['labels'][0],
                    'confidence': result['scores'][0]
                })
            
            return topics
            
        except Exception as e:
            logging.error(f"Error classifying topics: {str(e)}")
            return []

    def _find_similar_questions(self, questions):
        """Find groups of similar questions using sentence embeddings"""
        try:
            # Get embeddings for all questions
            texts = [q['text'] for q in questions]
            embeddings = self._model_context['sentence_transformer'].encode(texts)
            
            # Calculate similarity matrix
            similarities = np.inner(embeddings, embeddings)
            
            # Group similar questions (similarity > 0.8)
            similar_groups = {}
            for i in range(len(questions)):
                group = []
                for j in range(len(questions)):
                    if i != j and similarities[i][j] > 0.8:
                        group.append(j)
                if group:
                    similar_groups[i] = group
                
            return similar_groups
        except Exception as e:
            logging.error(f"Error finding similar questions: {str(e)}")
            return {}

    def _calculate_question_frequencies(self, questions):
        """Calculate frequency of each question"""
        try:
            frequencies = {}
            texts = [q['text'] for q in questions]
            for text in texts:
                frequencies[text] = texts.count(text)
            return frequencies
        except Exception as e:
            logging.error(f"Error calculating frequencies: {str(e)}")
            return {}

    def __del__(self):
        """Cleanup method"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except:
            pass