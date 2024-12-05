export default function TopicList({ topics = [] }) {
    return (
      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Frequent Topics</h2>
        <div className="space-y-4">
          {topics.map((topic, index) => (
            <div key={index} className="flex items-center justify-between">
              <span className="text-gray-700">{topic.name}</span>
              <div className="flex items-center">
                <div className="w-32 bg-gray-200 rounded-full h-2 mr-2">
                  <div
                    className="bg-blue-500 rounded-full h-2"
                    style={{ width: `${topic.frequency}%` }}
                  />
                </div>
                <span className="text-sm text-gray-500">{topic.frequency}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }