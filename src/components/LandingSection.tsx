import { Brain, Shield, Zap, BarChart, Upload, Lock } from 'lucide-react';

interface LandingSectionProps {
  onGetStarted: () => void;
}

export default function LandingSection({ onGetStarted }: LandingSectionProps) {
  const features = [
    {
      icon: Shield,
      title: 'Advanced AI Detection',
      description: 'Utilizing cutting-edge neural networks to identify AI-generated content with high accuracy'
    },
    {
      icon: Zap,
      title: 'Real-Time Analysis',
      description: 'Get instant results with our high-performance detection engine'
    },
    {
      icon: BarChart,
      title: 'Detailed Reports',
      description: 'Comprehensive analysis breakdown with confidence scores and detection methods'
    },
    {
      icon: Lock,
      title: 'Privacy First',
      description: 'Your uploads are automatically deleted after analysis, ensuring complete privacy'
    }
  ];

  const stats = [
    { value: '99.8%', label: 'Detection Accuracy' },
    { value: '500K+', label: 'Images Analyzed' },
    { value: '<2s', label: 'Average Processing Time' }
  ];

  return (
    <div className="relative overflow-hidden bg-[#0a0a0a]">
      {/* Hero Section */}
      <div className="relative pt-16 pb-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <div className="flex justify-center mb-8">
              <div className="p-4 bg-green-500/10 rounded-2xl border border-green-500/20">
                <Brain className="w-16 h-16 text-green-400" />
              </div>
            </div>
            <h1 className="text-4xl tracking-tight font-extrabold sm:text-5xl md:text-6xl">
              <span className="block bg-gradient-to-r from-blue-400 to-purple-800 text-transparent bg-clip-text hover:from-blue-300 hover:to-purple-700 transition-all duration-300">Detect AI-Generated</span>
              <span className="block bg-gradient-to-r from-purple-800 to-blue-400 text-transparent bg-clip-text hover:from-purple-700 hover:to-blue-300 transition-all duration-300">Content Instantly</span>
            </h1>
            <p className="mt-3 max-w-md mx-auto text-base text-gray-400 sm:text-lg md:mt-5 md:text-xl md:max-w-3xl">
              Harness the power of advanced AI to detect artificially generated images and videos with industry-leading accuracy.
            </p>
            <div className="mt-8">
              <button
                onClick={onGetStarted}
                className="btn-primary inline-flex items-center space-x-2 bg-gradient-to-r from-green-500 to-blue-600 hover:from-green-400 hover:to-blue-500 transition-all duration-300 shadow-lg hover:shadow-green-500/25"
              >
                <Upload className="w-5 h-5 text-white" />
                <span className="text-white drop-shadow-md">Start Analyzing</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            {stats.map((stat, index) => (
              <div key={index} className="card p-6 text-center bg-gradient-to-br from-[#111111] to-[#1a1a1a] border border-green-500/20">
                <dt className="text-4xl font-extrabold text-green-400">{stat.value}</dt>
                <dd className="mt-2 text-sm text-blue-300">{stat.label}</dd>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="relative mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-800 text-transparent bg-clip-text">
              Advanced Features
            </h2>
            <p className="mt-4 text-gray-400">
              Our platform offers cutting-edge tools for accurate content analysis
            </p>
          </div>
          <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-4">
            {features.map((feature, index) => (
              <div key={index} className="card p-6 bg-gradient-to-br from-[#111111] to-[#1a1a1a] border border-green-500/20">
                <div className="p-3 bg-gradient-to-br from-green-500/10 to-blue-600/10 rounded-lg w-fit border border-green-500/20">
                  <feature.icon className="w-6 h-6 text-green-400" />
                </div>
                <h3 className="mt-4 text-lg font-medium bg-gradient-to-r from-blue-400 to-purple-800 text-transparent bg-clip-text">
                  {feature.title}
                </h3>
                <p className="mt-2 text-gray-400 text-sm">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}