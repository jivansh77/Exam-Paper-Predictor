from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class QuestionAnalyzer:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=1000)
        self.topics_cache = {}
        self.questions_cache = {}
        
    def analyze(self, processed_data):
        questions = processed_data['questions']
        # Ensure clusters is a list if it's a NumPy array
        clusters = processed_data['clusters']
        if isinstance(clusters, np.ndarray):
            clusters = clusters.tolist()
        
        question_types = processed_data['question_types']
        
        # Calculate topic frequencies
        topic_counts = Counter(clusters)
        total_questions = len(questions)
        
        topics = []
        for topic_id, count in topic_counts.items():
            frequency = (count / total_questions) * 100
            topic_questions = [q for i, q in enumerate(questions) if clusters[i] == topic_id]
            
            # Get representative terms for topic
            tfidf_matrix = self.tfidf.fit_transform(topic_questions)
            feature_names = self.tfidf.get_feature_names_out()
            topic_terms = self._get_top_terms(tfidf_matrix, feature_names)
            
            topics.append({
                'id': topic_id,
                'name': f"Topic {topic_id}: {', '.join(topic_terms[:3])}",
                'frequency': round(frequency, 2)
            })
        
        # Calculate question relevance scores
        question_data = []
        for i, question in enumerate(questions):
            relevance = self._calculate_relevance(question, topic_questions)
            question_data.append({
                'text': question,
                'type': question_types[i],
                'topic': topics[clusters[i]]['name'],
                'relevance': round(relevance * 100, 2)
            })
        
        # Prepare analysis results
        results = {
            'totalQuestions': total_questions,
            'theoryCount': question_types.count('theory'),
            'numericalCount': question_types.count('numerical'),
            'patterns': self._identify_patterns(questions, question_types)
        }
        
        return {
            'topics': topics,
            'questions': question_data,
            'results': results
        }
    
    def _get_top_terms(self, tfidf_matrix, feature_names, n=3):
        avg_weights = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
        top_indices = avg_weights.argsort()[-n:][::-1]
        return [feature_names[i] for i in top_indices]
    
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