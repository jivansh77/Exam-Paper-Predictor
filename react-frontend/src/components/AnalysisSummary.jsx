export default function AnalysisSummary({ results }) {
  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-4">Analysis Summary</h2>
      <div className="grid grid-cols-3 gap-4">
        <div className="text-center">
          <p className="text-gray-600">Total Questions</p>
          <p className="text-2xl font-bold">{results.totalQuestions}</p>
        </div>
        <div className="text-center">
          <p className="text-gray-600">Theory Questions</p>
          <p className="text-2xl font-bold">{results.theoryCount}</p>
        </div>
        <div className="text-center">
          <p className="text-gray-600">Numerical Questions</p>
          <p className="text-2xl font-bold">{results.numericalCount}</p>
        </div>
      </div>
      
      {results.patterns && results.patterns.length > 0 && (
        <div className="mt-6">
          <h3 className="font-semibold mb-2">Identified Patterns</h3>
          {results.patterns.map((pattern, index) => (
            <div key={index} className="bg-blue-50 p-3 rounded mb-2">
              {pattern}
            </div>
          ))}
        </div>
      )}
    </div>
  )
} 