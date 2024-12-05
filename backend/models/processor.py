import pdfplumber
import pytesseract
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import spacy
import re

class ExamPaperProcessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.tfidf = TfidfVectorizer(max_features=1000)
        self.doc2vec = Doc2Vec(vector_size=100, min_count=2, epochs=30)
        
    def extract_from_pdf(self, pdf_path):
        """Extract text from PDF, handling both text and image-based PDFs"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ''
                for page in pdf.pages:
                    # Try text extraction first
                    page_text = page.extract_text()
                    if not page_text.strip():
                        # If no text found, treat as image
                        img = page.to_image()
                        page_text = pytesseract.image_to_string(img.original)
                    text += page_text + '\n'
                return text
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    def extract_questions(self, text):
        """Extract questions using regex patterns and structural analysis"""
        # Common question patterns
        patterns = [
            r'Q\d+\..*?\n',  # Q1. format
            r'\d+\).*?\n',   # 1) format
            r'\d+\.\s.*?\n'  # 1. format
        ]
        
        questions = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                question = match.group().strip()
                if len(question) > 10:  # Filter out noise
                    questions.append(question)
        
        return questions

    def preprocess_text(self, text):
        """Clean and normalize text"""
        doc = self.nlp(text)
        
        # Basic cleaning
        tokens = [token.lemma_.lower() for token in doc 
                 if not token.is_stop and not token.is_punct 
                 and token.text.strip()]
        
        return ' '.join(tokens)

    def train_doc2vec(self, questions):
        """Train Doc2Vec model on questions"""
        tagged_data = [TaggedDocument(self.preprocess_text(q).split(), [i]) 
                      for i, q in enumerate(questions)]
        self.doc2vec.build_vocab(tagged_data)
        self.doc2vec.train(tagged_data, 
                          total_examples=self.doc2vec.corpus_count, 
                          epochs=self.doc2vec.epochs)
    
    def get_embeddings(self, questions):
        """Generate Doc2Vec embeddings"""
        return np.array([
            self.doc2vec.infer_vector(self.preprocess_text(q).split())
            for q in questions
        ])

    def cluster_topics(self, embeddings, n_clusters=5):
        """Cluster questions into topics using hierarchical clustering"""
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = clustering.fit_predict(embeddings)
        
        return clusters

    def classify_question_type(self, question):
        """Classify question as theory or numerical"""
        # Check for numerical indicators
        numerical_indicators = ['calculate', 'solve', 'find', 'compute', 'evaluate']
        doc = self.nlp(question.lower())
        
        # Check for numbers and mathematical symbols
        has_numbers = bool(re.search(r'\d', question))
        has_math_symbols = bool(re.search(r'[+\-*/=]', question))
        
        if has_numbers or has_math_symbols or any(ind in question.lower() for ind in numerical_indicators):
            return 'numerical'
        return 'theory'

    def process_paper(self, pdf_path):
        """Complete pipeline for processing a paper"""
        # Extract text
        raw_text = self.extract_from_pdf(pdf_path)
        
        # Extract questions
        questions = self.extract_questions(raw_text)
        
        # Preprocess questions
        processed_questions = [self.preprocess_text(q) for q in questions]
        
        # Generate embeddings
        embeddings = self.get_embeddings(processed_questions)
        
        # Cluster into topics
        clusters = self.cluster_topics(embeddings)
        
        # Classify questions
        question_types = [self.classify_question_type(q) for q in questions]
        
        return {
            'questions': questions,
            'clusters': clusters,
            'question_types': question_types,
            'embeddings': embeddings
        }