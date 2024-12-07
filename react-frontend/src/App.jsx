import { useState } from 'react'
import TopicDistribution from './components/TopicDistribution'
import AnalysisSummary from './components/AnalysisSummary'
import QuestionsList from './components/QuestionsList'
import Sidebar from './components/Sidebar'
import FileUpload from './components/FileUpload'

function App() {
  const [analysisData, setAnalysisData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileUpload = async (file) => {
    setLoading(true)
    setError(null)
    
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('http://127.0.0.1:5000/analyze', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Failed to analyze file')
      }

      const data = await response.json()
      setAnalysisData(data)
    } catch (err) {
      setError('Error processing the file. Please try again.')
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
          <div className="text-center py-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
            <p className="mt-2">Analyzing paper...</p>
          </div>
        )}

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            {error}
          </div>
        )}

        {analysisData && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8">
            <TopicDistribution topics={analysisData.topics} />
            <AnalysisSummary results={analysisData.results} />
            <div className="md:col-span-2">
              <QuestionsList questions={analysisData.questions} />
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
