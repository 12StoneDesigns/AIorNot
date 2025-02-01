import React from 'react';
import { History, Calendar, Download, Trash2, ExternalLink } from 'lucide-react';

// Mock history data (replace with actual data from backend)
const historyData = [
  {
    id: 1,
    date: '2024-03-15T10:30:00Z',
    imageUrl: 'https://images.unsplash.com/photo-1682687220742-aba13b6e50ba',
    result: { isAIGenerated: true, confidenceScore: 0.92 },
    filename: 'analysis_001.jpg',
    size: '2.4 MB'
  },
  {
    id: 2,
    date: '2024-03-14T15:45:00Z',
    imageUrl: 'https://images.unsplash.com/photo-1682687221038-404670f01c30',
    result: { isAIGenerated: false, confidenceScore: 0.15 },
    filename: 'analysis_002.jpg',
    size: '1.8 MB'
  }
];

export default function HistorySection() {
  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="card">
        <div className="p-6 border-b border-teal-500/20">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-br from-teal-500/20 to-blue-600/20 rounded-lg">
                <History className="w-6 h-6 text-teal-400" />
              </div>
              <h2 className="text-2xl font-bold bg-gradient-to-r from-teal-400 to-blue-500 text-transparent bg-clip-text">Analysis History</h2>
            </div>
            <button className="text-sm text-teal-400 hover:text-teal-300 transition-colors">
              Clear All
            </button>
          </div>
        </div>

        <div className="divide-y divide-teal-500/10">
          {historyData.map((item) => (
            <div key={item.id} className="p-6 hover:bg-gradient-to-br from-teal-500/5 to-blue-600/5 transition-colors">
              <div className="flex items-start justify-between">
                <div className="flex space-x-4">
                  <div className="relative flex-shrink-0">
                    <img
                      src={item.imageUrl}
                      alt={`Analysis ${item.id}`}
                      className="w-24 h-24 rounded-lg object-cover border border-teal-500/20"
                    />
                    <div className={`absolute top-2 right-2 w-3 h-3 rounded-full ${
                      item.result.isAIGenerated ? 'bg-red-500' : 'bg-teal-500'
                    }`} />
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <Calendar className="w-4 h-4 text-teal-400" />
                      <span className="text-sm text-teal-400">
                        {new Date(item.date).toLocaleDateString('en-US', {
                          year: 'numeric',
                          month: 'long',
                          day: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit'
                        })}
                      </span>
                    </div>
                    <p className="mt-1 text-sm font-medium text-teal-300">
                      {item.filename}
                    </p>
                    <p className="mt-1 text-sm text-gray-400">
                      Size: {item.size}
                    </p>
                    <div className="mt-2 flex items-center space-x-2">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        item.result.isAIGenerated
                          ? 'bg-red-500/10 text-red-400'
                          : 'bg-teal-500/10 text-teal-400'
                      }`}>
                        {item.result.isAIGenerated ? 'AI Generated' : 'Authentic'}
                      </span>
                      <span className="text-gray-500">•</span>
                      <span className="text-sm text-gray-400">{(item.result.confidenceScore * 100).toFixed(1)}% confidence</span>
                    </div>
                  </div>
                </div>
                <div className="flex space-x-2">
                  <button className="p-2 text-teal-400 hover:text-teal-300 rounded-full hover:bg-zinc-800">
                    <Download className="w-5 h-5" />
                  </button>
                  <button className="p-2 text-teal-400 hover:text-teal-300 rounded-full hover:bg-zinc-800">
                    <ExternalLink className="w-5 h-5" />
                  </button>
                  <button className="p-2 text-teal-400 hover:text-red-400 rounded-full hover:bg-zinc-800">
                    <Trash2 className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}