import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import client from '../api/client';
import { Activity, Lock, User, UserPlus } from 'lucide-react';
import { Link } from 'react-router-dom';

const Register = () => {
  const { register, handleSubmit, formState: { errors } } = useForm();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  const onSubmit = async (data) => {
    setIsLoading(true);
    setError('');
    try {
      await client.post('/auth/register/', {
        username: data.username,
        password: data.password
      });
      setSuccess(true);
    } catch (err) {
      setError(err.response?.data?.detail || 'Registration failed. Username might be taken.');
    } finally {
      setIsLoading(false);
    }
  };

  if (success) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center p-6">
        <div className="max-w-md w-full bg-white rounded-2xl shadow-xl border border-slate-200 p-10 text-center">
          <div className="inline-flex items-center justify-center p-3 bg-green-100 rounded-xl mb-4">
            <UserPlus className="text-green-600 h-8 w-8" />
          </div>
          <h1 className="text-3xl font-bold text-slate-900 tracking-tight">Account Created</h1>
          <p className="text-slate-500 mt-2 mb-8">Your clinical account has been successfully registered.</p>
          <Link to="/login" className="w-full inline-block py-3 px-4 rounded-lg shadow-sm text-sm font-bold text-white bg-indigo-600 hover:bg-indigo-700 transition-colors">
            Go to Login
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center p-6">
      <div className="max-w-md w-full bg-white rounded-2xl shadow-xl border border-slate-200 p-10">
        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center p-3 bg-indigo-100 rounded-xl mb-4">
            <Activity className="text-indigo-600 h-8 w-8" />
          </div>
          <h1 className="text-3xl font-bold text-slate-900 tracking-tight">Clinical Registration</h1>
          <p className="text-slate-500 mt-2">Create a new account to access the RA engine</p>
        </div>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-slate-700">Username</label>
            <div className="mt-1 relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <User className="h-5 w-5 text-slate-400" />
              </div>
              <input
                {...register('username', { required: true, minLength: 3 })}
                className="block w-full pl-10 pr-3 py-2 border border-slate-300 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 text-sm"
                placeholder="clinician_name"
              />
            </div>
            {errors.username && <p className="text-red-500 text-[10px] mt-1">Min 3 characters required</p>}
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700">Password</label>
            <div className="mt-1 relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Lock className="h-5 w-5 text-slate-400" />
              </div>
              <input
                type="password"
                {...register('password', { required: true, minLength: 6 })}
                className="block w-full pl-10 pr-3 py-2 border border-slate-300 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 text-sm"
                placeholder="••••••••"
              />
            </div>
            {errors.password && <p className="text-red-500 text-[10px] mt-1">Min 6 characters required</p>}
          </div>

          {error && <p className="text-red-500 text-xs text-center font-medium">{error}</p>}

          <button
            type="submit"
            disabled={isLoading}
            className="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-bold text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-slate-400 transition-colors"
          >
            {isLoading ? 'Creating Account...' : 'Register'}
          </button>
        </form>

        <div className="mt-8 pt-8 border-t border-slate-100 text-center">
          <p className="text-sm text-slate-500">
            Already have an account? <Link to="/login" className="text-indigo-600 font-bold hover:text-indigo-500">Sign In</Link>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Register;
