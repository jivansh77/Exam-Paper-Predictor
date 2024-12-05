export default function AnalysisResults({ results }) {
    if (!results) return null;
  
    return (
      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Analysis Summary</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-blue-50 rounded-lg">
            <h3 className="font-medium text-blue-800">Total Questions</h3>
            <p className="text-2xl font-bold text-blue-600">{results.totalQuestions}</p>
          </div>
          <div className="p-4 bg-green-50 rounded-lg">
            <h3 className="font-medium text-green-800">Theory Questions</h3>
            <p className="text-2xl font-bold text-green-600">{results.theoryCount}</p>
          </div>
          <div className="p-4 bg-purple-50 rounded-lg">
            <h3 className="font-medium text-purple-800">Numerical Questions</h3>
            <p className="text-2xl font-bold text-purple-600">{results.numericalCount}</p>
          </div>
        </div>
        
        <div className="mt-6">
          <h3 className="font-medium mb-2">Pattern Analysis</h3>
          <ul className="space-y-2">
            {results.patterns?.map((pattern, index) => (
              <li key={index} className="flex items-center text-gray-700">
                <svg className="w-4 h-4 mr-2 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" />
                </svg>
                {pattern}
              </li>
            ))}
          </ul>
        </div>
      </div>
    );
  }