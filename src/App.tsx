import React, { useState } from 'react';
import { SettingsProvider } from './contexts/SettingsContext';
import Header from './components/Header';
import UploadSection from './components/UploadSection';
import SettingsPanel from './components/SettingsPanel';
import HistorySection from './components/HistorySection';
import ProfileSection from './components/ProfileSection';
import LandingSection from './components/LandingSection';

function App() {
  const [currentView, setCurrentView] = useState<'landing' | 'upload' | 'settings' | 'history' | 'profile'>('landing');

  return (
    <SettingsProvider>
      <div className="min-h-screen bg-gray-950">
        <div className="min-h-screen backdrop-blur-sm">
          <Header 
            onSettingsClick={() => setCurrentView('settings')}
            onUploadClick={() => setCurrentView('upload')}
            onHistoryClick={() => setCurrentView('history')}
            onProfileClick={() => setCurrentView('profile')}
            currentView={currentView}
          />
          <main className="main-container">
            <div className="card p-8 bg-gradient-to-br from-gray-900 to-gray-950 border border-teal-500/20">
              {currentView === 'landing' && <LandingSection onGetStarted={() => setCurrentView('upload')} />}
              {currentView === 'upload' && <UploadSection />}
              {currentView === 'settings' && <SettingsPanel />}
              {currentView === 'history' && <HistorySection />}
              {currentView === 'profile' && <ProfileSection />}
            </div>
          </main>
        </div>
      </div>
    </SettingsProvider>
  );
}

export default App;