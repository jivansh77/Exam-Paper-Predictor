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

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
)

export default function TopicDistribution({ topics }) {
  const data = {
    labels: topics.map(topic => topic.name),
    datasets: [
      {
        label: 'Frequency (%)',
        data: topics.map(topic => topic.frequency),
        backgroundColor: 'rgba(59, 130, 246, 0.5)',
      }
    ]
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-4">Topic Distribution</h2>
      <Bar data={data} />
    </div>
  )
} 