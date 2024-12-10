import { useState } from 'react'

export default function QuestionsList({ questions = [], allQuestions = [], clusters = [] }) {
  const [activeTab, setActiveTab] = useState('all')
  const [selectedCluster, setSelectedCluster] = useState(null)

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          <button
            className={`${
              activeTab === 'all'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            } whitespace-nowrap pb-4 px-1 border-b-2 font-medium`}
            onClick={() => setActiveTab('all')}
          >
            All Questions ({allQuestions.length})
          </button>
          <button
            className={`${
              activeTab === 'repeated'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            } whitespace-nowrap pb-4 px-1 border-b-2 font-medium`}
            onClick={() => setActiveTab('repeated')}
          >
            Repeated Questions ({questions.length})
          </button>
          <button
            className={`${
              activeTab === 'clusters'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            } whitespace-nowrap pb-4 px-1 border-b-2 font-medium`}
            onClick={() => setActiveTab('clusters')}
          >
            Similar Questions ({clusters.length})
          </button>
        </nav>
      </div>

      {activeTab === 'all' && allQuestions.length === 0 && (
        <p className="text-gray-500 text-center">No questions found</p>
      )}

      {activeTab === 'all' && allQuestions.length > 0 && (
        <div className="space-y-4">
          {allQuestions.map((question, index) => (
            <div key={index} className="border rounded p-4">
              <div className="flex justify-between items-start mb-2">
                <div className="flex-1">
                  <p className="font-medium">{question.text || question.question}</p>
                  <p className="text-sm text-gray-500">Topic: {question.topic}</p>
                </div>
              </div>
              <div className="text-sm text-gray-500">
                Type: {question.type}
              </div>
            </div>
          ))}
        </div>
      )}

      {activeTab === 'repeated' && questions.length === 0 && (
        <p className="text-gray-500 text-center">No repeated questions found</p>
      )}

      {activeTab === 'repeated' && questions.length > 0 && (
        <div className="space-y-4">
          {questions.map((question, index) => (
            <div key={index} className="border rounded p-4">
              <div className="flex justify-between items-start mb-2">
                <div className="flex-1">
                  <p className="font-medium">{question.question}</p>
                  <p className="text-sm text-gray-500">Topic: {question.topic}</p>
                </div>
                <span className="ml-4 px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                  Frequency: {question.frequency}
                </span>
              </div>
              <div className="text-sm text-gray-500">
                Type: {question.type}
              </div>
            </div>
          ))}
        </div>
      )}

      {activeTab === 'clusters' && clusters.length > 0 && (
        <div className="space-y-4">
          {clusters.map((cluster, index) => (
            <div key={index} className="border rounded p-4">
              <div 
                className="cursor-pointer"
                onClick={() => setSelectedCluster(selectedCluster === index ? null : index)}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="flex-1">
                    <p className="font-medium">{cluster.main_question}</p>
                    <p className="text-sm text-gray-500">Topic: {cluster.topic}</p>
                  </div>
                  <span className="ml-4 px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                    Similar: {cluster.similar_questions.length}
                  </span>
                </div>
              </div>
              
              {selectedCluster === index && (
                <div className="mt-4 pl-4 border-l-2 border-gray-200 space-y-3">
                  {cluster.similar_questions.map((question, sIndex) => (
                    <div key={sIndex} className="text-sm">
                      <p>{question}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
} 