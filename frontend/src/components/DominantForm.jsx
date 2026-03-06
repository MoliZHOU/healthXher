import React from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';

const schema = z.object({
  age: z.number().min(18).max(120),
  gender: z.enum(['Male', 'Female']),
  waist_cm: z.number().min(30).max(200),
  height_cm: z.number().min(50).max(250),
  neutrophils: z.number().min(0),
  lymphocytes: z.number().min(0),
  smoking_status: z.enum(['Never', 'Former', 'Current']),
  fiber_consumption: z.number().min(0),
  physical_activity: z.enum(['Sedentary', 'Moderate', 'Vigorous']),
  drinking_status: z.enum(['Almost non-drinker', 'Light drinker', 'Moderate drinker', 'Heavy drinker']),
  hypertension: z.enum(['Normal', 'Hypertension']),
  diabetes: z.enum(['Normal', 'Diabetes']),
  hyperlipidemia: z.enum(['Normal', 'Hyperlipidemia']),
  bmi: z.number().optional().nullable(),

  // ADDED：
  postpartum_12m: z.boolean(),
  family_history: z.boolean(),
  small_joint_symmetry: z.boolean(),
  MenopauseStatus: z.boolean(),
  MorningStiffnessLong: z.boolean(),
  SymptomsDuration6Weeks: z.boolean(),
 
});

const DominantForm = ({ onSubmit, isLoading, className = '', onUseOldVersion = () => {} }) => {
  const { register, handleSubmit, formState: { errors } } = useForm({
    resolver: zodResolver(schema),
    defaultValues: {
      physical_activity: 'Sedentary',
      drinking_status: 'Almost non-drinker',
      hypertension: 'Normal',
      diabetes: 'Normal',
      hyperlipidemia: 'Normal',
      bmi: null,
      // ADDED
      postpartum_12m: false,
      family_history: false,
      small_joint_symmetry: false,
    }
  });

  const [language, setLanguage] = React.useState('English');

  return (
    <form
      onSubmit={handleSubmit(onSubmit)}
      className={`space-y-6 bg-white p-8 rounded-xl shadow-lg border border-slate-200 ${className}`}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <label className="text-sm font-medium text-slate-700">Language</label>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="text-sm rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
          >
            <option value="English">English</option>
            <option value="Swedish">Swedish</option>
            <option value="Other">Other</option>
          </select>
        </div>
        <button
          type="button"
          onClick={onUseOldVersion}
          className="inline-flex items-center justify-center text-base md:text-lg font-semibold text-indigo-700 hover:text-indigo-900 px-4 py-2 rounded-lg border border-indigo-200 bg-indigo-50 hover:bg-indigo-100 shadow-sm"
        >
          Accessible Mode
        </button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Demographics */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-slate-800 border-b pb-2">Demographics</h3>
          <div>
            <label className="block text-sm font-medium text-slate-700">Age (18-120)</label>
            <input type="number" {...register('age', { valueAsNumber: true })} className="mt-1 block w-full rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500" />
            {errors.age && <p className="text-red-500 text-xs mt-1">{errors.age.message}</p>}
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700">Gender</label>
            <select {...register('gender')} className="mt-1 block w-full rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500">
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700">Smoking Status</label>
            <select {...register('smoking_status')} className="mt-1 block w-full rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500">
              <option value="Never">Never</option>
              <option value="Former">Former</option>
              <option value="Current">Current</option>
            </select>
          </div>
        </div>
        {/* Biometrics Section */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-slate-800 border-b pb-2">Biometrics</h3>
          
          {/* 第一行：Waist 和 Height */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700">Waist (cm)</label>
              <input type="number" step="0.1" {...register('waist_cm', { valueAsNumber: true })} className="mt-1 block w-full rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500" />
              {errors.waist_cm && <p className="text-red-500 text-xs mt-1">{errors.waist_cm.message}</p>}
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700">Height (cm)</label>
              <input type="number" step="0.1" {...register('height_cm', { valueAsNumber: true })} className="mt-1 block w-full rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500" />
              {errors.height_cm && <p className="text-red-500 text-xs mt-1">{errors.height_cm.message}</p>}
            </div>
          </div>

          {/* Group C: Lab Results (Optional) - 移出上面的 grid 独立成行 */}
          <div className="mt-6 border-t pt-4 border-slate-200">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-bold text-indigo-700 uppercase tracking-wider">
                Blood Test Results <span className="text-xs font-normal text-slate-400 lowercase">(Optional)</span>
              </h3>
            </div>
            
            <div className="grid grid-cols-2 gap-4 bg-slate-50 p-4 rounded-lg">
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">Neutrophils (10⁹/L)</label>
                <input 
                  type="number" 
                  step="0.01" 
                  placeholder="e.g. 4.5"
                  {...register('neutrophils', { valueAsNumber: true })} 
                  className="block w-full rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 text-sm" 
                />
                {errors.neutrophils && <p className="text-red-500 text-[10px] mt-1">{errors.neutrophils.message}</p>}
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">Lymphocytes (10⁹/L)</label>
                <input 
                  type="number" 
                  step="0.01" 
                  placeholder="e.g. 1.5"
                  {...register('lymphocytes', { valueAsNumber: true })} 
                  className="block w-full rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 text-sm" 
                />
                {errors.lymphocytes && <p className="text-red-500 text-[10px] mt-1">{errors.lymphocytes.message}</p>}
              </div>
            </div>
          </div>
        </div>  
        
        {/* <div className="space-y-4">
          <h3 className="text-lg font-semibold text-slate-800 border-b pb-2">Biometrics</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700">Waist (cm)</label>
              <input type="number" step="0.1" {...register('waist_cm', { valueAsNumber: true })} className="mt-1 block w-full rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500" />
              {errors.waist_cm && <p className="text-red-500 text-xs mt-1">{errors.waist_cm.message}</p>}
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700">Height (cm)</label>
              <input type="number" step="0.1" {...register('height_cm', { valueAsNumber: true })} className="mt-1 block w-full rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500" />
              {errors.height_cm && <p className="text-red-500 text-xs mt-1">{errors.height_cm.message}</p>}
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            {/* <div>
              <label className="block text-sm font-medium text-slate-700">Neutrophils</label>
              <input type="number" step="0.01" {...register('neutrophils', { valueAsNumber: true })} className="mt-1 block w-full rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500" />
              {errors.neutrophils && <p className="text-red-500 text-xs mt-1">{errors.neutrophils.message}</p>}
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700">Lymphocytes</label>
              <input type="number" step="0.01" {...register('lymphocytes', { valueAsNumber: true })} className="mt-1 block w-full rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500" />
              {errors.lymphocytes && <p className="text-red-500 text-xs mt-1">{errors.lymphocytes.message}</p>}
            </div> */}
            
            {/* Group C: Lab Results (Optional) */}


        {/* RA Clinical Markers & Hormonal Context */}
        <div className="md:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-6 bg-slate-50 p-6 rounded-xl border border-slate-200 mt-6">
          
          {/* Group A: Clinical Symptoms (高权重症状) */}
          <div className="space-y-4">
            <h3 className="text-sm font-bold text-indigo-900 uppercase tracking-wider">Clinical Presentation</h3>
            <div className="space-y-3">
              <div className="flex items-center space-x-3">
                <input type="checkbox" {...register('SmallJointSymmetry')} className="h-4 w-4 text-indigo-600" />
                <label className="text-sm text-slate-700">Small Joint Symmetry </label>
              </div>
              <div className="flex items-center space-x-3">
                <input type="checkbox" {...register('MorningStiffnessLong')} className="h-4 w-4 text-indigo-600" />
                  <label className="text-sm text-slate-700">Morning Stiffness &gt; 60 min</label>
              </div>
              <div className="flex items-center space-x-3">
                <input type="checkbox" {...register('SymptomsDuration6Weeks')} className="h-4 w-4 text-indigo-600" />
                <label className="text-sm text-slate-700">Duration ≥ 6 Weeks </label>
              </div>
            </div>
          </div>

          {/* Group B: Bio-Background*/}
          <div className="space-y-4 border-l pl-6 border-slate-200">
            <h3 className="text-sm font-bold text-pink-700 uppercase tracking-wider">Physiological Context</h3>
            <div className="space-y-3">
              <div className="flex items-center space-x-3">
                <input type="checkbox" {...register('Postpartum_12m')} className="h-4 w-4 text-pink-600" />
                <label className="text-sm text-slate-700">Postpartum (Last 12 Months)</label>
              </div>
              <div className="flex items-center space-x-3">
                <input type="checkbox" {...register('MenopauseStatus')} className="h-4 w-4 text-pink-600" />
                <label className="text-sm text-slate-700">Menopause </label>
              </div>
              <div className="flex items-center space-x-3">
                <input type="checkbox" {...register('Family History')} className="h-4 w-4 text-slate-600" />
                <label className="text-sm text-slate-700">Family History</label>
              </div>
            </div>
          </div>
        </div>

        {/* Lifestyle & Health */}
        <div className="md:col-span-2 grid grid-cols-1 md:grid-cols-3 gap-4 border-t pt-4">
          <div>
            <label className="block text-sm font-medium text-slate-700">Fiber (g/day)</label>
            <input type="number" step="0.1" {...register('fiber_consumption', { valueAsNumber: true })} className="mt-1 block w-full rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500" />
            {errors.fiber_consumption && <p className="text-red-500 text-xs mt-1">{errors.fiber_consumption.message}</p>}
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700">Physical Activity</label>
            <select {...register('physical_activity')} className="mt-1 block w-full rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500">
              <option value="Sedentary">Sedentary</option>
              <option value="Moderate">Moderate</option>
              <option value="Vigorous">Vigorous</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700">Drinking Status</label>
            <select {...register('drinking_status')} className="mt-1 block w-full rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500">
              <option value="Almost non-drinker">Almost non-drinker</option>
              <option value="Light drinker">Light drinker</option>
              <option value="Moderate drinker">Moderate drinker</option>
              <option value="Heavy drinker">Heavy drinker</option>
            </select>
          </div>
        </div>
      </div>
      <div className="mt-6 mb-4 flex items-start space-x-3">
        <input 
          type="checkbox" 
          id="data-consent"
          className="h-4 w-4 mt-0.5 text-indigo-600 border-slate-300 rounded focus:ring-indigo-500 cursor-pointer" 
        />
        <label htmlFor="data-consent" className="text-xs text-slate-500 leading-normal cursor-pointer">
          I consent to the collection and anonymized analysis of my health data.(Optional: You can still analyze your profile without checking this box)
        </label>
      </div>



      <button
        type="submit"
        disabled={isLoading}
        className="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-slate-400"
      >
        {isLoading ? 'Calculating Risk...' : 'Analyze Health Profile'}
      </button>
    </form>
  );
};

export default DominantForm;
