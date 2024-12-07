import pdfplumber
import pytesseract
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import spacy
import re
import logging

class ExamPaperProcessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.tfidf = TfidfVectorizer(max_features=1000)
        self.doc2vec = None
        self.question_frequency = {}  # Track question frequency
        
    def extract_from_pdf(self, pdf_path):
        """Extract text from PDF with improved formatting preservation"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_chunks = []
                for page in pdf.pages:
                    page_text = page.extract_text(
                        x_tolerance=3,
                        y_tolerance=3
                    )
                    
                    if not page_text.strip():
                        img = page.to_image(resolution=300)
                        page_text = pytesseract.image_to_string(
                            img.original,
                            config='--psm 6'
                        )
                    
                    # page_text = self._clean_extracted_text(page_text)
                    text_chunks.append(page_text)
                
                final_text = '\n'.join(text_chunks)
                logging.info(f'Extracted {len(text_chunks)} pages of text')
                return final_text
        except Exception as e:
            logging.error(f"PDF extraction error: {str(e)}")
            raise

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
        # Create a new Doc2Vec model for each training session
        self.doc2vec = Doc2Vec(
            vector_size=100,
            window=5,
            min_count=2,
            workers=4,
            epochs=30
        )
        
        # Prepare tagged documents
        tagged_data = [TaggedDocument(words=self.preprocess_text(q).split(), tags=[i]) 
                      for i, q in enumerate(questions)]
        
        # Build vocabulary and train
        self.doc2vec.build_vocab(tagged_data)
        self.doc2vec.train(tagged_data, 
                          total_examples=self.doc2vec.corpus_count, 
                          epochs=self.doc2vec.epochs)
        
        logging.info('Doc2Vec model training completed')
    
    def get_embeddings(self, questions):
        """Generate Doc2Vec embeddings"""
        if self.doc2vec is None:
            raise ValueError("Doc2Vec model has not been trained. Call train_doc2vec first.")
            
        try:
            embeddings = np.array([
                self.doc2vec.infer_vector(self.preprocess_text(q).split())
                for q in questions
            ])
            if embeddings.size == 0:
                raise ValueError("No embeddings generated")
            return embeddings
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            raise

    def cluster_topics(self, embeddings, n_clusters=5):
        """Cluster questions into topics using hierarchical clustering"""
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = clustering.fit_predict(embeddings)
        
        return clusters

    def classify_question_type(self, question):
      """Classify question as theory or numerical with more sophisticated checks"""
      # Expanded list of numerical indicators
      numerical_indicators = [
          'calculate', 'solve', 'find', 'compute', 'evaluate', 
          'how much', 'how many', 'what is the value', 'determine',
          'what percentage', 'what proportion', 'estimate', 'quantify'
      ]
      
      # Expanded list of theory-related keywords
      theory_indicators = [
          'explain', 'describe', 'discuss', 'define', 'outline', 
          'compare', 'contrast', 'analyze', 'critically assess', 
          'evaluate the impact', 'what are the reasons', 'why',
          'how does', 'what is the significance', 'discuss the role',
          'what are the main characteristics'
      ]
      
      # Convert question to lowercase for easier matching
      lower_question = question.lower()
      
      # Comprehensive checks
      # 1. Check for explicit numerical indicators
      if any(ind in lower_question for ind in numerical_indicators):
          return 'numerical'
      
      # 2. Check for explicit theory indicators
      if any(ind in lower_question for ind in theory_indicators):
          return 'theory'
      
      # 3. Advanced mathematical symbol and number detection
      # More precise regular expressions
      math_symbol_pattern = r'[+\-*/=<>≤≥]'
      number_pattern = r'\b(\d+(\.\d+)?)\b'
      
      has_math_symbols = bool(re.search(math_symbol_pattern, question))
      has_numbers = bool(re.search(number_pattern, question))
      
      # 4. Check for specific numerical contexts
      numerical_contexts = [
          'percentage', 'ratio', 'proportion', 'fraction', 
          'average', 'mean', 'median', 'mode', 'standard deviation'
      ]
      has_numerical_context = any(context in lower_question for context in numerical_contexts)
      
      # Decision logic
      if has_math_symbols or (has_numbers and has_numerical_context):
          return 'numerical'
      elif has_numbers:
          # If numbers are present but don't seem computational, still classify carefully
          # Check sentence structure and surrounding context
          doc = self.nlp(question)
          
          # Look at the dependency parsing
          numerical_deps = ['nummod', 'quantmod']
          has_numerical_dependency = any(
              token.dep_ in numerical_deps for token in doc
          )
          
          return 'numerical' if has_numerical_dependency else 'theory'
      
      # Default to theory if no clear numerical indicators
      return 'theory'

    def process_paper(self, pdf_path):
        """Complete pipeline for processing a paper"""
        # Extract text
        raw_text = self.extract_from_pdf(pdf_path)
        logging.info(f'Raw text extracted: {raw_text}...')

        # Extract questions
        questions = self.extract_questions(raw_text)
        logging.info(f'Questions extracted: {len(questions)} questions')

        if not questions:
            raise Exception("No questions found in the extracted text.")

        # Train Doc2Vec model
        self.train_doc2vec(questions)

        # Generate embeddings
        embeddings = self.get_embeddings(questions)
        logging.info(f'Embeddings shape: {embeddings.shape}')

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

    def process_multiple_papers(self, pdf_paths):
        all_questions = []
        all_embeddings = []
        
        for pdf_path in pdf_paths:
            paper_data = self.process_paper(pdf_path)
            
            # Track question frequency
            for question in paper_data['questions']:
                normalized_q = self.normalize_question(question)
                self.question_frequency[normalized_q] = int(self.question_frequency.get(normalized_q, 0) + 1)  # Convert to int
            
            all_questions.extend(paper_data['questions'])
            all_embeddings.extend(paper_data['embeddings'])
            
        # Cluster all questions together
        clusters = self.cluster_topics(np.array(all_embeddings))
        clusters = [int(c) for c in clusters]  # Convert to standard Python int
        
        return {
            'questions': all_questions,
            'clusters': clusters,
            'question_types': [self.classify_question_type(q) for q in all_questions],
            'frequencies': {k: int(v) for k, v in self.question_frequency.items()}  # Convert frequencies to int
        }
        
    def normalize_question(self, question):
        """Normalize question text to identify similar questions"""
        doc = self.nlp(question.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return ' '.join(sorted(tokens))