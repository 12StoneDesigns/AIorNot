import React from 'react';
import { User, Mail, Key, Shield, CreditCard, BarChart, Camera } from 'lucide-react';

const stats = [
  { label: 'Total Analyses', value: '247', icon: BarChart },
  { label: 'AI Detected', value: '142', icon: Shield },
  { label: 'Subscription', value: 'Pro', icon: CreditCard }
];

export default function ProfileSection() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="card bg-gradient-to-br from-[#0a0a0a] to-[#111111] border border-green-500/20">
        <div className="p-6">
          <div className="flex items-center space-x-6">
            <div className="relative">
              <div className="w-24 h-24 rounded-full bg-[#111111] flex items-center justify-center border border-green-500/20">
                <User className="w-12 h-12 text-green-400" />
              </div>
              <button className="absolute bottom-0 right-0 rounded-full bg-[#111111] p-1.5 shadow-sm border border-green-500/20 hover:bg-green-500/10 transition-colors">
                <Camera className="w-4 h-4 text-green-400" />
              </button>
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-300">John Doe</h2>
              <p className="text-sm text-gray-400 mt-1">john.doe@example.com</p>
              <div className="mt-4 flex space-x-3">
                <button className="btn-primary bg-gradient-to-r from-green-500 to-purple-600 hover:from-green-400 hover:to-purple-500">
                  Edit Profile
                </button>
                <button className="btn-secondary bg-[#111111] border border-green-500/20 hover:bg-green-500/10">
                  Change Password
                </button>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 divide-y sm:divide-y-0 sm:divide-x divide-green-500/20">
          {stats.map((stat) => (
            <div key={stat.label} className="p-6 text-center">
              <stat.icon className="mx-auto h-6 w-6 text-green-400" />
              <p className="mt-2 text-3xl font-bold text-gray-300">{stat.value}</p>
              <p className="mt-1 text-sm text-gray-400">{stat.label}</p>
            </div>
          ))}
        </div>

        <div className="p-6 border-t border-green-500/20">
          <h3 className="text-lg font-medium text-gray-300 mb-4">Account Security</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-[#111111] rounded-lg border border-green-500/20">
              <div className="flex items-center space-x-3">
                <Key className="w-5 h-5 text-green-400" />
                <div>
                  <p className="text-sm font-medium text-gray-300">Password</p>
                  <p className="text-xs text-gray-400">Last changed 3 months ago</p>
                </div>
              </div>
              <button className="text-sm text-green-400 hover:text-green-300">
                Change
              </button>
            </div>
            <div className="flex items-center justify-between p-4 bg-[#111111] rounded-lg border border-green-500/20">
              <div className="flex items-center space-x-3">
                <Shield className="w-5 h-5 text-green-400" />
                <div>
                  <p className="text-sm font-medium text-gray-300">Two-Factor Authentication</p>
                  <p className="text-xs text-gray-400">Not enabled</p>
                </div>
              </div>
              <button className="text-sm text-green-400 hover:text-green-300">
                Enable
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}