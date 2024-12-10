from collections import Counter
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass
import logging

@dataclass
class AnalysisResult:
    total_questions: int
    topic_distribution: Dict[str, float]
    common_patterns: List[str]
    question_clusters: List[Dict[str, Any]]
    repeated_questions: List[Dict[str, Any]]

class QuestionAnalyzer:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=1000)
        logging.getLogger().setLevel(logging.INFO)

    def analyze_papers(self, processor_output: Dict) -> Dict:
        """
        Analyze the processed exam papers data
        
        Args:
            processor_output: Output from ExamPaperProcessor containing questions, 
                            topics, similar groups, and frequencies
        """
        try:
            logging.info("Starting analysis...")
            questions = processor_output['questions']
            topics = processor_output['topics']  # This is now a list of strings
            similar_groups = processor_output['similar_groups']
            frequencies = processor_output['frequencies']

            # Calculate topic distribution
            topic_counts = Counter(q['topic'] for q in questions)  # Get topics from questions instead
            total = sum(topic_counts.values())
            topic_distribution = {
                topic: (count / total) * 100 
                for topic, count in topic_counts.items()
            }

            # Get top topics
            top_topics = dict(sorted(
                topic_distribution.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5])

            # Process question patterns
            patterns = self._identify_patterns(questions)

            # Process similar questions
            question_clusters = self._process_similar_groups(
                questions, similar_groups
            )

            # Get repeated questions
            repeated_questions = self._get_repeated_questions(
                questions, frequencies
            )

            analysis_result = {
                'total_questions': len(questions),
                'unique_topics': len(topic_counts),
                'topic_distribution': topic_distribution,
                'top_topics': top_topics,
                'patterns': patterns,
                'question_clusters': question_clusters,
                'repeated_questions': repeated_questions
            }

            logging.info("Analysis completed successfully")
            return analysis_result

        except Exception as e:
            logging.error(f"Error in analysis: {str(e)}", exc_info=True)
            raise

    def _identify_patterns(self, questions: List[Dict]) -> List[str]:
        """Identify patterns in questions"""
        patterns = []
        
        # Count question types
        type_counts = Counter(q['type'] for q in questions)
        total = len(questions)
        
        # Add type distribution pattern
        for qtype, count in type_counts.items():
            percentage = (count / total) * 100
            patterns.append(f"{percentage:.1f}% of questions are {qtype} type")
            
        return patterns

    def _process_similar_groups(self, questions: List[Dict], similar_groups: Dict) -> List[Dict]:
        """Process similar question groups"""
        clusters = []
        
        for main_idx, similar_indices in similar_groups.items():
            clusters.append({
                'main_question': questions[main_idx]['text'],
                'similar_questions': [questions[idx]['text'] for idx in similar_indices],
                'topic': questions[main_idx]['topic']
            })
            
        return clusters

    def _get_repeated_questions(self, questions: List[Dict], frequencies: Dict) -> List[Dict]:
        """Get frequently repeated questions"""
        repeated = []
        
        for q in questions:
            freq = frequencies.get(q['text'], 1)
            if freq > 1:  # Only include questions that appear more than once
                repeated.append({
                    'question': q['text'],
                    'topic': q['topic'],
                    'type': q['type'],
                    'frequency': freq,
                    'confidence': 1.0  # You might want to calculate this differently
                })
                
        # Sort by frequency
        repeated.sort(key=lambda x: x['frequency'], reverse=True)
        
        return repeated