import { useState } from 'react'
import TopicDistribution from './components/TopicDistribution'
import AnalysisSummary from './components/AnalysisSummary'
import QuestionsList from './components/QuestionsList'
import Sidebar from './components/Sidebar'
import FileUpload from './components/FileUpload'

export default function App() {
  const [analysisData, setAnalysisData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileUpload = async (files) => {
    setLoading(true)
    setError(null)
    
    try {
      const formData = new FormData()
      files.forEach(file => formData.append('files[]', file))
      
      const response = await fetch('http://127.0.0.1:5000/analyze', {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) {
        throw new Error('Analysis failed')
      }
      
      const data = await response.json()
      console.log('Received data:', data)
      
      const transformedData = {
        analysis: {
          topic_distribution: data.topic_distribution || {},
          all_questions: (data.questions || []).map(q => ({
            text: q.text,
            topic: q.topic,
            type: q.type
          })),
          repeated_questions: data.repeated_questions || [],
          question_clusters: data.similar_groups || []
        },
        summary: {
          total_questions: data.total_questions || 0,
          unique_topics: data.unique_topics || 0,
          top_topics: Object.fromEntries(
            Object.entries(data.topic_distribution || {})
              .sort(([,a], [,b]) => b - a)
              .slice(0, 5)
          ),
          patterns: data.patterns || []
        }
      }
      
      console.log('Transformed data:', transformedData)
      setAnalysisData(transformedData)
    } catch (err) {
      setError(err.message)
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen bg-gray-100">
      <Sidebar />
      
      <main className="flex-1 p-8">
        <h1 className="text-3xl font-bold mb-8">Exam Paper Analyzer</h1>
        
        <FileUpload onUpload={handleFileUpload} />
        
        {loading && (
          <div className="mt-8 text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
            <p className="mt-4 text-gray-600">Analyzing papers...</p>
          </div>
        )}
        
        {error && (
          <div className="mt-8 p-4 bg-red-100 text-red-700 rounded">
            {error}
          </div>
        )}
        
        {analysisData && (
          <div className="mt-8 space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <TopicDistribution 
                topics={analysisData.analysis.topic_distribution} 
                questions={analysisData.analysis.repeated_questions}
              />
              <AnalysisSummary summary={analysisData.summary} />
            </div>
            <QuestionsList 
              questions={analysisData.analysis.repeated_questions}
              allQuestions={analysisData.analysis.all_questions}
              clusters={analysisData.analysis.question_clusters} 
            />
          </div>
        )}
      </main>
    </div>
  )
}
