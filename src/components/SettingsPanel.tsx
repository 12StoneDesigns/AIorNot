import React, { useState, useEffect } from 'react';
import { Settings, Sliders, Bell, Lock, Database, Share2 } from 'lucide-react';

interface SettingsOption {
  id: string;
  label: string;
  description: string;
  value: boolean;
}

export default function SettingsPanel() {
  const [activeTab, setActiveTab] = useState('analysis');
  const [settings, setSettings] = useState({
    analysis: [
      {
        id: 'deepAnalysis',
        label: 'Deep Analysis Mode',
        description: 'Enable comprehensive analysis with higher accuracy but slower processing',
        value: false
      },
      {
        id: 'autoDetect',
        label: 'Auto-Detection',
        description: 'Automatically start analysis when images are uploaded',
        value: true
      },
      {
        id: 'saveHistory',
        label: 'Save Analysis History',
        description: 'Store analysis results for future reference',
        value: true
      }
    ],
    notifications: [
      {
        id: 'emailNotifications',
        label: 'Email Notifications',
        description: 'Receive analysis results via email',
        value: false
      },
      {
        id: 'browserNotifications',
        label: 'Browser Notifications',
        description: 'Show desktop notifications when analysis is complete',
        value: true
      }
    ],
    privacy: [
      {
        id: 'anonymousMode',
        label: 'Anonymous Mode',
        description: 'Do not store any personal data or upload history',
        value: false
      },
      {
        id: 'autoDelete',
        label: 'Auto-Delete Results',
        description: 'Automatically delete analysis results after 24 hours',
        value: false
      }
    ]
  });

  // Load settings from localStorage on component mount
  useEffect(() => {
    const savedSettings = localStorage.getItem('analysisSettings');
    if (savedSettings) {
      const parsedSettings = JSON.parse(savedSettings);
      setSettings(prev => ({
        ...prev,
        analysis: prev.analysis.map(item => ({
          ...item,
          value: item.id === 'deepAnalysis' ? parsedSettings.deepAnalysis :
                 item.id === 'autoDetect' ? parsedSettings.autoDetect :
                 item.id === 'saveHistory' ? parsedSettings.saveHistory :
                 item.value
        }))
      }));
    }
  }, []);

  const handleToggle = (section: string, id: string) => {
    setSettings(prev => {
      const newSettings = {
        ...prev,
        [section]: prev[section as keyof typeof prev].map(item =>
          item.id === id ? { ...item, value: !item.value } : item
        )
      };
      
      // Save to localStorage
      if (section === 'analysis') {
        const analysisSettings = {
          deepAnalysis: newSettings.analysis.find(item => item.id === 'deepAnalysis')?.value || false,
          autoDetect: newSettings.analysis.find(item => item.id === 'autoDetect')?.value || true,
          saveHistory: newSettings.analysis.find(item => item.id === 'saveHistory')?.value || true
        };
        localStorage.setItem('analysisSettings', JSON.stringify(analysisSettings));
      }
      
      return newSettings;
    });
  };

  const renderToggle = (option: SettingsOption, section: string) => (
    <div key={option.id} className="flex items-start space-x-4 p-4 hover:bg-zinc-800/50 rounded-lg transition-all duration-200">
      <div className="flex-1">
        <div className="flex items-center justify-between">
          <label htmlFor={option.id} className="font-medium text-zinc-100">
            {option.label}
          </label>
          <button
            id={option.id}
            type="button"
            className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-zinc-500 focus:ring-offset-2 focus:ring-offset-zinc-900 ${
              option.value ? 'bg-zinc-600' : 'bg-zinc-700'
            }`}
            onClick={() => handleToggle(section, option.id)}
          >
            <span
              className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-zinc-100 shadow ring-0 transition duration-200 ease-in-out ${
                option.value ? 'translate-x-5' : 'translate-x-0'
              }`}
            />
          </button>
        </div>
        <p className="mt-1 text-sm text-zinc-400">{option.description}</p>
      </div>
    </div>
  );

  return (
    <div className="bg-zinc-900 rounded-xl shadow-2xl overflow-hidden max-w-4xl mx-auto border border-zinc-800">
      <div className="flex items-center justify-between p-6 border-b border-zinc-800 bg-zinc-900/50">
        <div className="flex items-center space-x-2">
          <Settings className="w-6 h-6 text-zinc-400" />
          <h2 className="text-2xl font-bold text-zinc-100">Settings</h2>
        </div>
      </div>

      <div className="flex">
        {/* Sidebar */}
        <div className="w-64 border-r border-zinc-800 bg-zinc-900/50 p-4">
          <div className="space-y-1">
            {[
              { id: 'analysis', icon: Sliders, label: 'Analysis' },
              { id: 'notifications', icon: Bell, label: 'Notifications' },
              { id: 'privacy', icon: Lock, label: 'Privacy' },
              { id: 'api', icon: Database, label: 'API' },
              { id: 'sharing', icon: Share2, label: 'Sharing' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 w-full px-3 py-2 text-sm font-medium rounded-md transition-all duration-200 ${
                  activeTab === tab.id
                    ? 'bg-gradient-to-r from-zinc-800 to-zinc-700 text-zinc-100 shadow-lg'
                    : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-300'
                }`}
              >
                <tab.icon className="w-5 h-5" />
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Main content */}
        <div className="flex-1 p-6 bg-zinc-900">
          <div className="space-y-6">
            {activeTab === 'analysis' && (
              <div>
                <h3 className="text-lg font-medium text-zinc-100 mb-4">Analysis Settings</h3>
                <div className="space-y-2">
                  {settings.analysis.map(option => renderToggle(option, 'analysis'))}
                </div>
              </div>
            )}

            {activeTab === 'notifications' && (
              <div>
                <h3 className="text-lg font-medium text-zinc-100 mb-4">Notification Preferences</h3>
                <div className="space-y-2">
                  {settings.notifications.map(option => renderToggle(option, 'notifications'))}
                </div>
              </div>
            )}

            {activeTab === 'privacy' && (
              <div>
                <h3 className="text-lg font-medium text-zinc-100 mb-4">Privacy Settings</h3>
                <div className="space-y-2">
                  {settings.privacy.map(option => renderToggle(option, 'privacy'))}
                </div>
              </div>
            )}

            {activeTab === 'api' && (
              <div>
                <h3 className="text-lg font-medium text-zinc-100 mb-4">API Configuration</h3>
                <div className="space-y-4">
                  <div className="rounded-lg border border-zinc-800 bg-zinc-800/50 p-4">
                    <label className="block text-sm font-medium text-zinc-300 mb-2">API Key</label>
                    <div className="flex gap-2">
                      <input
                        type="password"
                        className="flex-1 rounded-lg border-zinc-700 bg-zinc-900 text-zinc-100 focus:border-zinc-600 focus:ring-zinc-600 text-sm px-3 py-2"
                        placeholder="Enter your API key"
                      />
                      <button className="px-4 py-2 bg-gradient-to-r from-zinc-700 to-zinc-600 hover:from-zinc-600 hover:to-zinc-500 text-zinc-100 rounded-lg transition-all duration-200 shadow-lg font-medium">
                        Save Changes
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'sharing' && (
              <div>
                <h3 className="text-lg font-medium text-zinc-100 mb-4">Sharing Options</h3>
                <div className="space-y-4">
                  <div className="rounded-lg bg-zinc-800/50 p-4 border border-zinc-800">
                    <p className="text-sm text-zinc-400">
                      Configure how analysis results can be shared with others.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}