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
});

const DominantForm = ({ onSubmit, isLoading }) => {
  const { register, handleSubmit, formState: { errors } } = useForm({
    resolver: zodResolver(schema),
    defaultValues: {
      physical_activity: 'Sedentary',
      drinking_status: 'Almost non-drinker',
      hypertension: 'Normal',
      diabetes: 'Normal',
      hyperlipidemia: 'Normal',
      bmi: null,
    }
  });

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6 bg-white p-8 rounded-xl shadow-lg border border-slate-200">
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

        {/* Biometrics */}
        <div className="space-y-4">
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
            <div>
              <label className="block text-sm font-medium text-slate-700">Neutrophils</label>
              <input type="number" step="0.01" {...register('neutrophils', { valueAsNumber: true })} className="mt-1 block w-full rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500" />
              {errors.neutrophils && <p className="text-red-500 text-xs mt-1">{errors.neutrophils.message}</p>}
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700">Lymphocytes</label>
              <input type="number" step="0.01" {...register('lymphocytes', { valueAsNumber: true })} className="mt-1 block w-full rounded-md border-slate-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500" />
              {errors.lymphocytes && <p className="text-red-500 text-xs mt-1">{errors.lymphocytes.message}</p>}
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
