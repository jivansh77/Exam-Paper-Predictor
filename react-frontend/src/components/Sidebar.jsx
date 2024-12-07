import React from 'react'

export default function Sidebar() {
  return (
    <aside className="w-64 bg-white shadow-lg min-h-screen p-6">
      <div className="mb-8">
        <h2 className="text-xl font-bold mb-4">About</h2>
        <p className="text-gray-600">
          Upload exam papers to analyze common topics and questions.
        </p>
      </div>

      <div>
        <h2 className="text-xl font-bold mb-4">Instructions</h2>
        <ol className="list-decimal list-inside space-y-3 text-gray-600">
          <li>Upload a PDF file of past exam papers</li>
          <li>Wait for analysis to complete</li>
          <li>View topic distribution and common questions</li>
        </ol>
      </div>
    </aside>
  )
} 