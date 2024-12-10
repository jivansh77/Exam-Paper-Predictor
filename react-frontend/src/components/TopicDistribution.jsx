import { Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'
import { useState } from 'react'

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
)

export default function TopicDistribution({ topics, questions }) {
  const [expandedTopic, setExpandedTopic] = useState(null);
  
  // Group questions by topic
  const questionsByTopic = questions?.reduce((acc, q) => {
    if (!acc[q.topic]) {
      acc[q.topic] = [];
    }
    acc[q.topic].push(q);
    return acc;
  }, {}) || {};

  const data = {
    labels: Object.keys(topics),
    datasets: [{
      label: 'Topic Distribution (%)',
      data: Object.values(topics),
      backgroundColor: 'rgba(59, 130, 246, 0.5)',
    }]
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: 'Topic Distribution' }
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow space-y-6">
      <Bar options={options} data={data} />
      
      <div className="mt-6 space-y-4">
        <h3 className="font-semibold text-lg">Questions by Topic</h3>
        {Object.entries(questionsByTopic).map(([topic, topicQuestions]) => (
          <div key={topic} className="border rounded-lg">
            <button
              className="w-full px-4 py-2 text-left flex justify-between items-center hover:bg-gray-50"
              onClick={() => setExpandedTopic(expandedTopic === topic ? null : topic)}
            >
              <span className="font-medium">{topic}</span>
              <span className="text-gray-500">
                {topicQuestions.length} questions
              </span>
            </button>
            
            {expandedTopic === topic && (
              <div className="p-4 border-t space-y-3">
                {topicQuestions.map((q, idx) => (
                  <div key={idx} className="p-3 bg-gray-50 rounded">
                    <p>{q.text}</p>
                    <p className="text-sm text-gray-500 mt-1">
                      Type: {q.type}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
} 