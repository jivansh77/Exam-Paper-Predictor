export default function AnalysisSummary({ summary }) {
  if (!summary) {
    return null;
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-4">Analysis Summary</h2>
      
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="text-center p-4 bg-blue-50 rounded">
          <p className="text-gray-600">Total Questions</p>
          <p className="text-2xl font-bold">{summary.total_questions || 0}</p>
        </div>
        <div className="text-center p-4 bg-blue-50 rounded">
          <p className="text-gray-600">Unique Topics</p>
          <p className="text-2xl font-bold">{summary.unique_topics || 0}</p>
        </div>
      </div>
      
      {summary.top_topics && (
        <div className="mb-6">
          <h3 className="font-semibold mb-2">Top Topics</h3>
          <div className="space-y-2">
            {Object.entries(summary.top_topics).map(([topic, percentage]) => (
              <div key={topic} className="flex justify-between items-center">
                <span>{topic}</span>
                <span className="font-semibold">
                  {typeof percentage === 'number' ? percentage.toFixed(1) : '0'}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {summary.patterns && summary.patterns.length > 0 && (
        <div>
          <h3 className="font-semibold mb-2">Identified Patterns</h3>
          <ul className="space-y-2">
            {summary.patterns.map((pattern, index) => (
              <li key={index} className="bg-gray-50 p-2 rounded">
                {pattern}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
} 