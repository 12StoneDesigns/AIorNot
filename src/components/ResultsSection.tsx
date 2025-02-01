import React from 'react';
import { X } from 'lucide-react';
import './ResultsSection.css';

interface ResultsSectionProps {
  result: {
    label: string;
    score: number;
    metadata?: {
      resolution?: string;
      format?: string;
      size?: string;
      analysisMode?: string;
      confidence?: {
        [key: string]: number;
      };
    };
  };
  mediaUrl: string;
  onClose: () => void;
}

const ResultsSection: React.FC<ResultsSectionProps> = ({ result, mediaUrl, onClose }) => {
  const isAIGenerated = result.label.toLowerCase().includes('ai');

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 rounded-xl w-full max-w-3xl max-h-[90vh] overflow-y-auto">
        <div className={`p-6 relative ${
          isAIGenerated 
            ? 'bg-gradient-to-b from-rose-500/20 to-transparent' 
            : 'bg-gradient-to-b from-emerald-500/20 to-transparent'
        }`}>
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div className={`text-2xl font-semibold text-white`}>
                {result.label}
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-gray-800/50 transition-colors"
            >
              <X className="w-5 h-5 text-gray-400" />
            </button>
          </div>

          {/* Analysis Details */}
          <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className={`p-4 rounded-lg border ${
              isAIGenerated
                ? 'bg-red-500/5 border-red-500/20'
                : 'bg-emerald-500/5 border-emerald-500/20'
            }`}>
              <h3 className={`text-lg font-semibold mb-2 text-white`}>
                Score Breakdown
              </h3>
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-300">Overall Confidence</span>
                    <span className="text-sm font-semibold text-white">{(result.score * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 w-full bg-gray-800/50 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${
                        isAIGenerated 
                          ? 'bg-gradient-to-r from-red-500 to-red-600' 
                          : 'bg-gradient-to-r from-emerald-500 to-emerald-600'
                      }`}
                      style={{ width: `${result.score * 100}%` }}
                    />
                  </div>
                </div>
                <div className="text-sm text-gray-300">
                  <p className="mb-2">
                    <span className="font-semibold text-white">Score Interpretation:</span>
                  </p>
                  <ul className="space-y-2 list-disc pl-4">
                    <li>
                      <span className="text-white font-medium">Below 45%:</span>{' '}
                      {isAIGenerated 
                        ? "Few AI patterns detected, suggesting possible human made content"
                        : "Limited human made characteristics found"}
                    </li>
                    <li>
                      <span className="text-white font-medium">45% - 65%:</span>{' '}
                      {isAIGenerated 
                        ? "Moderate AI characteristics present"
                        : "Mixed patterns detected"}
                    </li>
                    <li>
                      <span className="text-white font-medium">Above 65%:</span>{' '}
                      {isAIGenerated 
                        ? "Strong AI patterns identified, high confidence in AI generation"
                        : "Clear human made patterns found, including natural variations and organic irregularities"}
                    </li>
                  </ul>
                </div>
              </div>
            </div>
            <div className={`p-4 rounded-lg border ${
              isAIGenerated
                ? 'bg-red-500/5 border-red-500/20'
                : 'bg-emerald-500/5 border-emerald-500/20'
            }`}>
              <h3 className={`text-lg font-semibold mb-2 text-white`}>
                Pattern Analysis
              </h3>
              <div className="text-sm text-gray-300 space-y-4">
                <p>
                  {isAIGenerated 
                    ? "Our system has detected characteristics typically associated with AI-generated content:"
                    : "Our analysis has identified patterns typically found in human made content:"}
                </p>
                <ul className="list-disc pl-4 space-y-2">
                  {isAIGenerated ? (
                    <>
                      <li>Consistent patterns in structure and formatting</li>
                      <li>Repetitive or predictable elements</li>
                      <li>Uniform characteristics throughout the content</li>
                      <li>Machine-optimized patterns</li>
                    </>
                  ) : (
                    <>
                      <li>Natural variations in style and structure</li>
                      <li>Organic irregularities and imperfections</li>
                      <li>Unique creative elements</li>
                      <li>Subtle inconsistencies typical of human made content</li>
                    </>
                  )}
                </ul>
              </div>
            </div>
          </div>

          {/* Media Preview */}
          <div className="mt-8">
            <h3 className="text-lg font-medium mb-4 text-white">Media Preview</h3>
            <div className="rounded-lg overflow-hidden bg-gray-800/30">
              <img 
                src={mediaUrl} 
                alt="Analysis media"
                className="w-full h-auto object-contain"
                onError={(e) => {
                  const target = e.target as HTMLImageElement;
                  target.style.display = 'none';
                }}
              />
            </div>
          </div>

          {/* Media Information */}
          {result.metadata && (
            <div className="mt-8">
              <h3 className="text-lg font-medium mb-4 text-white">Media Information</h3>
              <div className="grid grid-cols-2 gap-4">
                {Object.entries(result.metadata).map(([key, value]) => (
                  value && key !== 'confidence' && (
                    <div key={key} className="bg-gray-800/30 rounded-lg p-4">
                      <div className="text-white text-sm capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}</div>
                      <div className="text-white mt-1">{typeof value === 'string' ? value : JSON.stringify(value)}</div>
                    </div>
                  )
                ))}
              </div>
            </div>
          )}
        </div>

        <div className={`sticky bottom-0 p-6 border-t border-gray-800`}>
          <div className="flex justify-end">
            <button
              onClick={onClose}
              className={`px-6 py-2 rounded-lg transition-all duration-300 font-medium ${
                isAIGenerated
                  ? 'bg-rose-500 hover:bg-rose-600 text-white'
                  : 'bg-emerald-500 hover:bg-emerald-600 text-white'
              }`}
            >
              Close Analysis
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsSection;