export default function QuestionList({ questions = [] }) {
    return (
      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Common Questions</h2>
        <div className="space-y-4">
          {questions.map((question, index) => (
            <div key={index} className="border-l-4 border-blue-500 pl-4 py-2">
              <div className="flex justify-between items-start">
                <p className="text-gray-800">{question.text}</p>
                <span className={`ml-4 px-2 py-1 rounded text-sm ${
                  question.type === 'theory' ? 'bg-green-100 text-green-800' : 'bg-purple-100 text-purple-800'
                }`}>
                  {question.type}
                </span>
              </div>
              <div className="mt-2 text-sm text-gray-500">
                Relevance: {question.relevance}% | Topic: {question.topic}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }