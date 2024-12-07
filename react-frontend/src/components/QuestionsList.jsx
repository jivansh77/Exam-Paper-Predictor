import { useState } from 'react'

export default function QuestionsList({ questions }) {
  const [questionType, setQuestionType] = useState('All')
  const [minRelevance, setMinRelevance] = useState(50)

  const filteredQuestions = questions.filter(q => 
    (questionType === 'All' || q.type.toLowerCase() === questionType.toLowerCase()) &&
    q.relevance >= minRelevance
  )

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-4">Common Questions</h2>
      
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700">Filter by Type</label>
          <select
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            value={questionType}
            onChange={(e) => setQuestionType(e.target.value)}
          >
            <option>All</option>
            <option>Theory</option>
            <option>Numerical</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Minimum Relevance: {minRelevance}%
          </label>
          <input
            type="range"
            min="0"
            max="100"
            value={minRelevance}
            onChange={(e) => setMinRelevance(Number(e.target.value))}
            className="mt-1 block w-full"
          />
        </div>
      </div>

      <div className="space-y-4">
        {filteredQuestions.map((question, index) => (
          <div key={index} className="border rounded-lg p-4">
            <h3 className="font-medium mb-2">{question.text}</h3>
            <div className="grid grid-cols-3 gap-4 text-sm text-gray-600">
              <div>Type: {question.type}</div>
              <div>Topic: {question.topic}</div>
              <div>Relevance: {question.relevance}%</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
} 