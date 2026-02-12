import React, { useState } from 'react';
import client from '../api/client';
import DominantForm from '../components/DominantForm';
import { PieChart, Pie, Cell, ResponsiveContainer, Text } from 'recharts';
import { Activity, Database, Camera, LogOut } from 'lucide-react';

const RiskGauge = ({ probability }) => {
  const data = [
    { name: 'Risk', value: probability * 100 },
    { name: 'Remaining', value: 100 - (probability * 100) },
  ];
  const COLORS = [probability > 0.5 ? '#ef4444' : '#f59e0b', '#e2e8f0'];

  return (
    <div className="h-64 w-full relative flex items-center justify-center">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="100%"
            startAngle={180}
            endAngle={0}
            innerRadius={80}
            outerRadius={120}
            paddingAngle={0}
            dataKey="value"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      <div className="absolute bottom-0 text-center">
        <span className="text-4xl font-bold text-slate-800">{(probability * 100).toFixed(1)}%</span>
        <p className="text-slate-500 font-medium">Risk Probability</p>
      </div>
    </div>
  );
};

const Dashboard = () => {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handlePredict = async (data) => {
    setIsLoading(true);
    setError('');
    try {
      const response = await client.post('/predict/', data);
      setResult(response.data);
    } catch (err) {
      const detail = err.response?.data?.detail;
      if (Array.isArray(detail)) {
        // FastAPI validation errors are an array of objects
        setError(detail.map(d => `${d.loc.join('.')}: ${d.msg}`).join(', '));
      } else {
        setError(detail || 'An error occurred during prediction.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    window.location.href = '/login';
  };

  return (
    <div className="min-h-screen bg-slate-50">
      <nav className="bg-white border-b border-slate-200 px-6 py-4 flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <Activity className="text-indigo-600 h-8 w-8" />
          <span className="text-xl font-bold text-slate-900 tracking-tight">healthXher</span>
        </div>
        <button onClick={handleLogout} className="flex items-center space-x-1 text-slate-600 hover:text-red-600 transition-colors">
          <LogOut size={20} />
          <span className="text-sm font-medium">Logout</span>
        </button>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
          {/* Input Section */}
          <div className="lg:col-span-2 space-y-8">
            <div>
              <h1 className="text-3xl font-bold text-slate-900">Health Risk Assessment</h1>
              <p className="text-slate-600 mt-2">Enter your biometric and lifestyle data for a real-time clinical risk analysis.</p>
            </div>
            
            <DominantForm onSubmit={handlePredict} isLoading={isLoading} />
            
            {error && (
              <div className="bg-red-50 border-l-4 border-red-400 p-4 rounded">
                <p className="text-red-700">{error}</p>
              </div>
            )}
          </div>

          {/* Results & Roadmap Section */}
          <div className="space-y-8">
            <div className="bg-white p-8 rounded-xl shadow-lg border border-slate-200">
              <h2 className="text-xl font-bold text-slate-900 mb-6">Risk Analysis</h2>
              {result ? (
                <div className="space-y-8">
                  <RiskGauge probability={result.probability} />
                  <div className={`p-4 rounded-lg text-center ${result.probability >= 0.35 ? 'bg-red-100 text-red-800' : 'bg-amber-100 text-amber-800'}`}>
                    <span className="font-bold uppercase tracking-wider">{result.risk_level}</span>
                  </div>
                  {result.needs_followup && (
                    <p className="text-sm text-slate-600 italic text-center">
                      * Clinically significant risk detected. Please consult a healthcare professional.
                    </p>
                  )}
                </div>
              ) : (
                <div className="h-64 flex flex-col items-center justify-center text-slate-400">
                  <Database size={48} className="mb-4 opacity-20" />
                  <p>Awaiting Data Input...</p>
                </div>
              )}
            </div>

            {/* Future Features (GrayScale) */}
            <div className="bg-slate-100 p-8 rounded-xl border border-slate-200 grayscale opacity-60">
              <h2 className="text-xl font-bold text-slate-400 mb-6">Future Modules</h2>
              <div className="space-y-4">
                <div className="flex items-center space-x-4 p-4 bg-white rounded-lg border border-slate-200">
                  <Camera className="text-slate-400" />
                  <div>
                    <p className="font-bold text-slate-500">MRI Upload</p>
                    <p className="text-xs text-slate-400">Computer Vision Analysis</p>
                  </div>
                </div>
                <div className="flex items-center space-x-4 p-4 bg-white rounded-lg border border-slate-200">
                  <Database className="text-slate-400" />
                  <div>
                    <p className="font-bold text-slate-500">Hospital Sync</p>
                    <p className="text-xs text-slate-400">Direct EMR Integration</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
