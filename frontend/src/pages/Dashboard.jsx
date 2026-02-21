import React, { useEffect, useRef, useState } from 'react';
import client from '../api/client';
import DominantForm from '../components/DominantForm';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { Activity, Database, Camera, LogOut } from 'lucide-react';
import BodyImage from '../assets/body.jpg';
import { buildReportHtml } from '../utils/reportTemplate';

const USE_MOCK_PREDICT = true;
const HIGH_RISK_THRESHOLD = 0.35;

const clamp01 = (value) => Math.max(0, Math.min(1, value));

const getRiskLevel = (probability) => {
  if (probability >= 0.60) return 'Very High Risk (>60%)';
  if (probability >= 0.35) return 'High Risk (35-60%)';
  if (probability >= 0.15) return 'Moderate Risk (15-35%)';
  return 'Low Risk (<15%)';
};

const mockPredict = (data) => {
  const ageScore = clamp01((data.age - 18) / 102) * 0.25;
  const nlr = data.neutrophils && data.lymphocytes
    ? data.neutrophils / Math.max(data.lymphocytes, 0.1)
    : 1.5;
  const nlrScore = clamp01((nlr - 1.0) / 5.0) * 0.2;
  const smokingScore = data.smoking_status === 'Current'
    ? 0.15
    : data.smoking_status === 'Former'
      ? 0.08
      : 0.0;
  const fiberScore = clamp01(1 - (data.fiber_consumption / 30)) * 0.1;
  const activityScore = data.physical_activity === 'Vigorous'
    ? -0.05
    : data.physical_activity === 'Moderate'
      ? -0.02
      : 0.03;
  const comorbidityScore =
    (data.hypertension === 'Hypertension' ? 0.08 : 0) +
    (data.diabetes === 'Diabetes' ? 0.08 : 0) +
    (data.hyperlipidemia === 'Hyperlipidemia' ? 0.06 : 0);
  const painScore = (data.joint_pain ?? 0) / 100 * 0.12;
  const swellingScore = (data.joint_swelling ?? 0) / 100 * 0.12;

  const probability = clamp01(
    0.08 +
    ageScore +
    nlrScore +
    smokingScore +
    fiberScore +
    activityScore +
    comorbidityScore +
    painScore +
    swellingScore
  );
  return {
    probability,
    risk_level: getRiskLevel(probability),
    needs_followup: probability >= 0.15
  };
};

const JOINTS = [
  { id: 'shoulder_left', label: 'Left Shoulder', x: 30.39, y: 19.29 },
  { id: 'shoulder_right', label: 'Right Shoulder', x: 64.15, y: 19.88 },
  { id: 'elbow_left', label: 'Left Elbow', x: 24.94, y: 34.21 },
  { id: 'elbow_right', label: 'Right Elbow', x: 69.84, y: 34.21 },
  { id: 'wrist_left', label: 'Left Wrist', x: 17.75, y: 47.14 },
  { id: 'wrist_right', label: 'Right Wrist', x: 77.03, y: 47.19 },
  { id: 'hand_left', label: 'Left Hand', x: 11.28, y: 54.05 },
  { id: 'hand_right', label: 'Right Hand', x: 81.02, y: 54.69 },
  { id: 'hip_left', label: 'Left Hip', x: 37.24, y: 47.19 },
  { id: 'hip_right', label: 'Right Hip', x: 57.77, y: 47.19 },
  { id: 'knee_left', label: 'Left Knee', x: 37.01, y: 69.54 },
  { id: 'knee_right', label: 'Right Knee', x: 57.89, y: 69.54 },
  { id: 'ankle_left', label: 'Left Ankle', x: 37.01, y: 87.49 },
  { id: 'ankle_right', label: 'Right Ankle', x: 58.0, y: 87.55 },
  { id: 'toes_left', label: 'Left Toes', x: 34.45, y: 96.21 },
  { id: 'toes_right', label: 'Right Toes', x: 59.63, y: 96.37 }
];

const JointPainSwellingModal = ({ open, onClose, joint, scores, onChange }) => {
  const painRef = useRef(null);
  const swellingRef = useRef(null);
  const [dragging, setDragging] = useState(null);

  if (!open || !joint) return null;

  const pain = scores?.pain ?? 0;
  const swelling = scores?.swelling ?? 0;

  const updateFromEvent = (event, type) => {
    const ref = type === 'pain' ? painRef : swellingRef;
    if (!ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    const x = clamp01((event.clientX - rect.left) / rect.width);
    const value = Math.round(x * 100);
    if (type === 'pain') {
      onChange({ pain: value, swelling });
    } else {
      onChange({ pain, swelling: value });
    }
  };

  const handlePointerDown = (event, type) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(type);
    updateFromEvent(event, type);
  };

  const handlePointerMove = (event, type) => {
    if (dragging !== type) return;
    updateFromEvent(event, type);
  };

  const handlePointerUp = () => {
    setDragging(null);
  };

  const painPos = `${clamp01(pain / 100) * 100}%`;
  const swellingPos = `${clamp01(swelling / 100) * 100}%`;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-6">
      <div className="w-full max-w-2xl rounded-2xl bg-white shadow-2xl border border-slate-200 animate-modal-pop">
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100">
          <div>
            <h3 className="text-lg font-bold text-slate-900">{joint.label}</h3>
            <p className="text-sm text-slate-500">Drag to score pain and swelling.</p>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg px-3 py-1 text-sm font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-100"
          >
            Close
          </button>
        </div>
        <div className="p-6 space-y-6">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
              <div className="text-xs uppercase tracking-wide text-slate-500">Pain</div>
              <div className="text-2xl font-bold text-slate-900">{pain}</div>
            </div>
            <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
              <div className="text-xs uppercase tracking-wide text-slate-500">Swelling</div>
              <div className="text-2xl font-bold text-slate-900">{swelling}</div>
            </div>
          </div>
          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between text-xs text-slate-500 mb-2">
                <span>Pain</span>
                <span>High</span>
              </div>
              <div
                ref={painRef}
                onPointerDown={(event) => handlePointerDown(event, 'pain')}
                onPointerMove={(event) => handlePointerMove(event, 'pain')}
                onPointerUp={handlePointerUp}
                className="relative h-14 w-full rounded-full border border-slate-200 bg-gradient-to-r from-emerald-100 via-amber-100 to-rose-200 cursor-ew-resize touch-none overflow-hidden"
              >
                <div className="absolute inset-0 heat-layer" />
                <div
                  className="absolute top-[55%] h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-slate-900 shadow-lg pulse-dot"
                  style={{ left: painPos }}
                />
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between text-xs text-slate-500 mb-2">
                <span>Swelling</span>
                <span>High</span>
              </div>
              <div
                ref={swellingRef}
                onPointerDown={(event) => handlePointerDown(event, 'swelling')}
                onPointerMove={(event) => handlePointerMove(event, 'swelling')}
                onPointerUp={handlePointerUp}
                className="relative h-14 w-full rounded-full border border-slate-200 bg-gradient-to-r from-emerald-100 via-sky-100 to-indigo-200 cursor-ew-resize touch-none overflow-hidden"
              >
                <div className="absolute inset-0 heat-layer" />
                <div
                  className="absolute top-[55%] h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-slate-900 shadow-lg pulse-dot"
                  style={{ left: swellingPos }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const HighRiskReportModal = ({ open, onClose, onSave }) => {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-6">
      <div className="w-full max-w-3xl rounded-2xl bg-white shadow-2xl border border-slate-200 animate-modal-pop">
        <div className="flex items-start justify-between px-6 py-4 border-b border-slate-100">
          <div>
            <h3 className="text-lg font-bold text-slate-900">High Risk Clinical Report</h3>
            <p className="text-sm text-slate-500">Review the report and save a local copy.</p>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg px-3 py-1 text-sm font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-100"
          >
            Close
          </button>
        </div>
        <div className="px-6 py-5 space-y-4">
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-700">
            This will open a print-ready PDF report in a new tab with your latest patient data and recommendations.
            Choose “Save as PDF” in the print dialog to download a copy.
          </div>
          <div className="flex items-center justify-end gap-3">
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-600 hover:text-slate-900 hover:bg-slate-100"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={onSave}
              className="rounded-lg bg-rose-600 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-rose-700 transition-colors"
            >
              Save as PDF
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

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
  const [jointScores, setJointScores] = useState({});
  const [selectedJointId, setSelectedJointId] = useState(null);
  const [isJointModalOpen, setIsJointModalOpen] = useState(false);
  const [showHighRiskReport, setShowHighRiskReport] = useState(false);
  const [lastSubmittedData, setLastSubmittedData] = useState(null);

  const selectedJoint = JOINTS.find((joint) => joint.id === selectedJointId);
  const selectedScores = selectedJointId ? jointScores[selectedJointId] : null;
  const scoreEntries = Object.values(jointScores);
  const aggregatePain = scoreEntries.length
    ? Math.round(scoreEntries.reduce((sum, entry) => sum + entry.pain, 0) / scoreEntries.length)
    : 0;
  const aggregateSwelling = scoreEntries.length
    ? Math.round(scoreEntries.reduce((sum, entry) => sum + entry.swelling, 0) / scoreEntries.length)
    : 0;

  const handleJointClick = (jointId) => {
    setSelectedJointId(jointId);
    setIsJointModalOpen(true);
  };

  const handleJointChange = (values) => {
    if (!selectedJointId) return;
    setJointScores((prev) => ({
      ...prev,
      [selectedJointId]: {
        pain: values.pain ?? prev[selectedJointId]?.pain ?? 0,
        swelling: values.swelling ?? prev[selectedJointId]?.swelling ?? 0
      }
    }));
  };

  const getMarkerStyle = (jointId) => {
    const score = jointScores[jointId];
    if (!score) {
      return {};
    }
    const pain = score?.pain ?? 0;
    const swelling = score?.swelling ?? 0;
    const intensity = clamp01((pain + swelling) / 200);
    return {
      boxShadow: `0 0 0 6px rgba(239, 68, 68, ${0.12 + intensity * 0.35})`
    };
  };

  const handlePredict = async (data) => {
    setIsLoading(true);
    setError('');
    setLastSubmittedData(data);
    try {
      if (USE_MOCK_PREDICT) {
        await new Promise((resolve) => setTimeout(resolve, 400));
        const payload = {
          ...data,
          joint_pain: aggregatePain,
          joint_swelling: aggregateSwelling
        };
        setResult(mockPredict(payload));
        return;
      }
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

  useEffect(() => {
    if (!result || result.probability < HIGH_RISK_THRESHOLD) {
      setShowHighRiskReport(false);
    }
  }, [result]);

  const patientName = localStorage.getItem('username') || 'Unknown';
  const reportDate = new Date().toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: '2-digit'
  });
  const reportIdDate = new Date().toISOString().slice(0, 10).replace(/-/g, '');
  const reportId = `HR-${reportIdDate}`;
  const reportFieldLabels = {
    age: 'Age',
    gender: 'Gender',
    smoking_status: 'Smoking Status',
    waist_cm: 'Waist (cm)',
    height_cm: 'Height (cm)',
    neutrophils: 'Neutrophils',
    lymphocytes: 'Lymphocytes',
    fiber_consumption: 'Fiber (g/day)',
    physical_activity: 'Physical Activity',
    drinking_status: 'Drinking Status'
  };
  const reportGroups = [
    { title: 'Demographics', fields: ['age', 'gender', 'smoking_status'] },
    { title: 'Biometrics', fields: ['waist_cm', 'height_cm', 'neutrophils', 'lymphocytes'] },
    { title: 'Lifestyle & Health', fields: ['fiber_consumption', 'physical_activity', 'drinking_status'] }
  ];
  const getReportValue = (key) => {
    const value = lastSubmittedData ? lastSubmittedData[key] : undefined;
    if (value === null || value === undefined || value === '') return 'N/A';
    return value;
  };
  const reportValues = Object.fromEntries(
    Object.keys(reportFieldLabels).map((key) => [key, getReportValue(key)])
  );

  const handleSaveReport = () => {
    const reportWindow = window.open('', '_blank');
    if (!reportWindow) return;
    reportWindow.document.open();
    reportWindow.document.write(buildReportHtml({
      patientName,
      reportDate,
      reportId,
      probability: result ? (result.probability * 100).toFixed(1) : 'N/A',
      ringAngle: result ? Math.round(result.probability * 360) : 0,
      reportGroups,
      reportFieldLabels,
      reportValues,
      joints: JOINTS,
      jointScores,
    }));
    reportWindow.document.close();
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
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-slate-900">Health Risk Assessment</h1>
          <p className="text-slate-600 mt-2">Enter your biometric and lifestyle data for a real-time clinical risk analysis.</p>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 lg:items-stretch">
          {/* Joint Map */}
          <div className="lg:col-span-4 xl:col-span-3 space-y-6 h-full">
            <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200 h-full flex flex-col">
              <div className="flex flex-col gap-4">
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">Joint Symptom Map</h2>
                  <p className="text-sm text-slate-500">Tap a joint to score pain and swelling.</p>
                </div>
                <div className="flex flex-wrap items-center gap-3 text-sm">
                  <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                    Avg Pain <span className="font-semibold text-slate-900">{aggregatePain}</span>
                  </div>
                  <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
                    Avg Swelling <span className="font-semibold text-slate-900">{aggregateSwelling}</span>
                  </div>
                </div>
              </div>
              <div className="mt-4 w-full max-w-[360px] overflow-visible rounded-xl border border-slate-200 bg-slate-50 relative mx-auto lg:mx-0">
                <img src={BodyImage} alt="Body map" className="w-full h-auto object-contain rounded-xl" />
                {JOINTS.map((joint) => (
                  <button
                    key={joint.id}
                    type="button"
                    onClick={() => handleJointClick(joint.id)}
                    className={`joint-marker group${jointScores[joint.id] ? ' joint-marker-active' : ''}`}
                    style={{ left: `${joint.x}%`, top: `${joint.y}%`, ...getMarkerStyle(joint.id) }}
                  >
                    <span className="joint-dot" />
                    <span className="joint-label">{joint.label}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Input Section */}
          <div className="lg:col-span-5 xl:col-span-6 h-full">
            <div className="h-full space-y-4">
              <DominantForm onSubmit={handlePredict} isLoading={isLoading} className="h-full flex flex-col" />

              {error && (
                <div className="bg-red-50 border-l-4 border-red-400 p-4 rounded">
                  <p className="text-red-700">{error}</p>
                </div>
              )}
            </div>
          </div>

          {/* Results & Roadmap Section */}
          <div className="lg:col-span-3 xl:col-span-3 space-y-8">
            <div className="bg-white p-8 rounded-xl shadow-lg border border-slate-200">
              <h2 className="text-xl font-bold text-slate-900 mb-6">Risk Analysis</h2>
              {result ? (
                <div className="space-y-8">
                  <RiskGauge probability={result.probability} />
                  <div className={`p-4 rounded-lg text-center ${result.probability >= 0.35 ? 'bg-red-100 text-red-800' : 'bg-amber-100 text-amber-800'}`}>
                    <span className="font-bold uppercase tracking-wider">{result.risk_level}</span>
                  </div>
                  {result.probability >= HIGH_RISK_THRESHOLD && (
                    <div className="space-y-4 rounded-xl border border-rose-200 bg-rose-50/60 p-4">
                      <div className="space-y-2">
                        <p className="text-sm text-rose-900 font-semibold">High risk detected. Generate a clinical report?</p>
                        <p className="text-xs text-rose-700">We will prepare a detailed report with next-step testing and care guidance.</p>
                      </div>
                      <button
                        type="button"
                        onClick={() => setShowHighRiskReport(true)}
                        className="w-full rounded-lg bg-rose-600 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-rose-700 transition-colors"
                      >
                        Generate Report
                      </button>
                    </div>
                  )}
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

      <JointPainSwellingModal
        open={isJointModalOpen}
        onClose={() => setIsJointModalOpen(false)}
        joint={selectedJoint}
        scores={selectedScores}
        onChange={handleJointChange}
      />
      <HighRiskReportModal
        open={result?.probability >= HIGH_RISK_THRESHOLD && showHighRiskReport}
        onClose={() => setShowHighRiskReport(false)}
        onSave={handleSaveReport}
      />
    </div>
  );
};

export default Dashboard;
