import { useState } from 'react'

export default function QuestionsList({ questions }) {
  const [questionType, setQuestionType] = useState('All')
  const [minFrequency, setMinFrequency] = useState(1)
  const [expandedQuestion, setExpandedQuestion] = useState(null)

  const filteredQuestions = questions?.filter(q => 
    (questionType === 'All' || q.type.toLowerCase() === questionType.toLowerCase()) &&
    q.frequency >= minFrequency
  ) || []

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
            Minimum Frequency: {minFrequency}
          </label>
          <input
            type="range"
            min="1"
            max="10"
            value={minFrequency}
            onChange={(e) => setMinFrequency(Number(e.target.value))}
            className="mt-1 block w-full"
          />
        </div>
      </div>

      <div className="space-y-4">
        {filteredQuestions.map((question, index) => (
          <div 
            key={index}
            className="border rounded-lg p-4 hover:bg-gray-50 cursor-pointer"
            onClick={() => setExpandedQuestion(expandedQuestion === index ? null : index)}
          >
            <div className="flex justify-between items-start">
              <div className="flex-1">
                <p className="font-medium text-gray-900">
                  {question.text.length > 100 
                    ? `${question.text.substring(0, 100)}...` 
                    : question.text}
                </p>
                {expandedQuestion === index && (
                  <div className="mt-4 space-y-2">
                    <p className="text-gray-600"><span className="font-medium">Full Question:</span> {question.text}</p>
                    <p className="text-gray-600"><span className="font-medium">Type:</span> {question.type}</p>
                    <p className="text-gray-600">
                      <span className="font-medium">Frequency:</span> 
                      {question.frequency} {question.frequency === 1 ? 'time' : 'times'}
                    </p>
                  </div>
                )}
              </div>
              <div className="ml-4 flex-shrink-0">
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                  ${question.frequency > 1 ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}`}>
                  {question.frequency}×
                </span>
              </div>
            </div>
          </div>
        ))}

        {filteredQuestions.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            No questions match the current filters
          </div>
        )}
      </div>
    </div>
  )
} 