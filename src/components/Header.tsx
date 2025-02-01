import { Brain, Upload, History, User, Settings } from 'lucide-react';

interface HeaderProps {
  onSettingsClick: () => void;
  onUploadClick: () => void;
  onHistoryClick: () => void;
  onProfileClick: () => void;
  currentView: string;
}

export default function Header({ 
  onSettingsClick, 
  onUploadClick, 
  onHistoryClick, 
  onProfileClick,
  currentView 
}: HeaderProps) {
  return (
    <header className="bg-[#0a0a0a] border-b border-green-500/20 backdrop-blur-sm sticky top-0 z-50">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gradient-to-br from-green-500/10 to-blue-600/10 rounded-lg border border-green-500/20">
              <Brain className="w-6 h-6 text-green-400" />
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-800 text-transparent bg-clip-text">AIorNot</span>
          </div>
          <div className="hidden md:block">
            <div className="flex items-center space-x-2">
              <button
                onClick={onUploadClick}
                className={`nav-item ${currentView === 'upload' ? 'bg-gradient-to-r from-green-500/10 to-blue-600/10 border-green-500/20' : ''}`}
              >
                <Upload className="w-4 h-4 mr-2 inline-block text-green-400" />
                <span className="text-gray-400">Analyze</span>
              </button>
              <button
                onClick={onHistoryClick}
                className={`nav-item ${currentView === 'history' ? 'bg-gradient-to-r from-green-500/10 to-blue-600/10 border-green-500/20' : ''}`}
              >
                <History className="w-4 h-4 mr-2 inline-block text-green-400" />
                <span className="text-gray-400">History</span>
              </button>
              <button
                onClick={onProfileClick}
                className={`nav-item ${currentView === 'profile' ? 'bg-gradient-to-r from-green-500/10 to-blue-600/10 border-green-500/20' : ''}`}
              >
                <User className="w-4 h-4 mr-2 inline-block text-green-400" />
                <span className="text-gray-400">Profile</span>
              </button>
              <button
                onClick={onSettingsClick}
                className={`nav-item ${currentView === 'settings' ? 'bg-gradient-to-r from-green-500/10 to-blue-600/10 border-green-500/20' : ''}`}
              >
                <Settings className="w-4 h-4 mr-2 inline-block text-green-400" />
                <span className="text-gray-400">Settings</span>
              </button>
            </div>
          </div>
        </div>
      </nav>
    </header>
  );
}