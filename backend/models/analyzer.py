import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class QuestionAnalyzer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.tfidf = TfidfVectorizer(max_features=1000)
        self.topics_cache = {}
        self.questions_cache = {}
        
    def normalize_question(self, question):
        """Normalize question text to identify similar questions"""
        doc = self.nlp(question.lower())
        # Remove stop words and punctuation, lemmatize the remaining tokens
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return ' '.join(sorted(tokens))  # Sort tokens for consistent comparison
        
    def analyze(self, processed_data):
        questions = processed_data['questions']
        clusters = [int(c) for c in processed_data['clusters']]
        question_types = processed_data['question_types']
        frequencies = processed_data.get('frequencies', {})
        
        # Group questions by topic
        topic_questions = {}
        for i, cluster in enumerate(clusters):
            if cluster not in topic_questions:
                topic_questions[cluster] = []
            topic_questions[cluster].append({
                'text': questions[i],
                'frequency': int(frequencies.get(self.normalize_question(questions[i]), 1)),
                'type': question_types[i]
            })
        
        # Sort questions by frequency within each topic
        for topic in topic_questions:
            topic_questions[topic].sort(key=lambda x: x['frequency'], reverse=True)
        
        # Calculate topic frequencies
        topic_counts = Counter(clusters)
        total_questions = len(questions)
        
        topics = []
        for topic_id, count in topic_counts.items():
            frequency = float((count / total_questions) * 100)
            topic_questions_list = [q for i, q in enumerate(questions) if clusters[i] == topic_id]
            
            # Get representative terms for topic
            tfidf_matrix = self.tfidf.fit_transform(topic_questions_list)
            feature_names = self.tfidf.get_feature_names_out()
            topic_terms = self._get_top_terms(tfidf_matrix, feature_names)
            
            topics.append({
                'id': int(topic_id),
                'name': f"Topic {topic_id}: {', '.join(topic_terms[:3])}",
                'frequency': round(float(frequency), 2),
                'questions': topic_questions[topic_id]
            })
            
        # Calculate overall statistics
        results = {
            'totalQuestions': int(len(questions)),
            'theoryCount': int(question_types.count('theory')),
            'numericalCount': int(question_types.count('numerical')),
            'patterns': self._identify_patterns(questions, question_types)
        }
        
        return {
            'topics': topics,
            'results': results,
            'questions': [q for topic in topic_questions.values() for q in topic]
        }
    
    def _get_top_terms(self, tfidf_matrix, feature_names, n=3):
        """Get the top terms for a topic based on TF-IDF scores"""
        avg_weights = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
        top_indices = avg_weights.argsort()[-n:][::-1]
        return [str(feature_names[i]) for i in top_indices]
    
    def _calculate_relevance(self, question, topic_questions):
        if not topic_questions:
            return 0
        
        question_vector = self.tfidf.transform([question])
        topic_vectors = self.tfidf.transform(topic_questions)
        similarities = cosine_similarity(question_vector, topic_vectors)
        return similarities.mean()
    
    def _identify_patterns(self, questions, types):
        patterns = []
        
        # Theory vs Numerical distribution
        theory_ratio = types.count('theory') / len(types)
        if theory_ratio > 0.7:
            patterns.append("Strong focus on theoretical questions")
        elif theory_ratio < 0.3:
            patterns.append("Strong focus on numerical problems")
        
        # Question length patterns
        lengths = [len(q.split()) for q in questions]
        avg_length = sum(lengths) / len(lengths)
        if avg_length > 30:
            patterns.append("Questions tend to be detailed and lengthy")
        elif avg_length < 15:
            patterns.append("Questions are typically concise")
            
        return patterns
    
    def get_topics(self):
        return list(self.topics_cache.values())
    
    def get_questions_by_topic(self, topic_id):
        return self.questions_cache.get(topic_id, [])