import { useState } from 'react';
import FileUpload from './components/FileUpload';
import TopicList from './components/TopicList';
import QuestionList from './components/QuestionList';
import AnalysisResults from './components/AnalysisResults';

export default function App() {
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileUpload = async (file) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      // TODO: Replace with actual API endpoint
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      setAnalysisData(data);
    } catch (error) {
      console.error('Error analyzing file:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        <h1 className="text-3xl font-bold text-center mb-8">
          Exam Paper Predictor
        </h1>
        
        <FileUpload onFileUpload={handleFileUpload} />
        
        {loading && (
          <div className="text-center mt-8">
            <div className="animate-spin h-8 w-8 mx-auto border-4 border-blue-500 border-t-transparent rounded-full" />
            <p className="mt-2 text-gray-600">Analyzing paper...</p>
          </div>
        )}
        
        {analysisData && (
          <div className="mt-8 space-y-8">
            <TopicList topics={analysisData.topics} />
            <QuestionList questions={analysisData.questions} />
            <AnalysisResults results={analysisData.results} />
          </div>
        )}
      </div>
    </div>
  );
}