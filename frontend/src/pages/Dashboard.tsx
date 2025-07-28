import React from 'react';
import { useQuery } from 'react-query';
import {
  CubeIcon,
  CogIcon,
  DocumentTextIcon,
  ChartBarIcon,
  PlayIcon,
  ArrowUpIcon,
  ArrowDownIcon,
} from '@heroicons/react/24/outline';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

import { api } from '../services/api';

interface StatsCardProps {
  title: string;
  value: string | number;
  change?: number;
  icon: React.ComponentType<any>;
  color: string;
}

const StatsCard: React.FC<StatsCardProps> = ({ title, value, change, icon: Icon, color }) => (
  <div className="bg-white overflow-hidden shadow rounded-lg">
    <div className="p-5">
      <div className="flex items-center">
        <div className="flex-shrink-0">
          <Icon className={`h-6 w-6 ${color}`} />
        </div>
        <div className="ml-5 w-0 flex-1">
          <dl>
            <dt className="text-sm font-medium text-gray-500 truncate">{title}</dt>
            <dd className="text-lg font-medium text-gray-900">{value}</dd>
          </dl>
        </div>
      </div>
    </div>
    {change !== undefined && (
      <div className="bg-gray-50 px-5 py-3">
        <div className="text-sm">
          <div className="flex items-center">
            {change >= 0 ? (
              <ArrowUpIcon className="h-4 w-4 text-green-500" />
            ) : (
              <ArrowDownIcon className="h-4 w-4 text-red-500" />
            )}
            <span className={`ml-2 font-medium ${change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {Math.abs(change)}%
            </span>
            <span className="ml-2 text-gray-500">from last month</span>
          </div>
        </div>
      </div>
    )}
  </div>
);

const Dashboard: React.FC = () => {
  const { data: stats, isLoading } = useQuery('dashboard-stats', async () => {
    // Mock data - replace with actual API call
    return {
      totalModels: 12,
      activeTraining: 3,
      totalDatasets: 8,
      totalEvaluations: 24,
      totalInferences: 156,
      feedbackCollected: 89,
    };
  });

  const trainingData = [
    { name: 'Jan', ppo: 4, dpo: 2, a2c: 1 },
    { name: 'Feb', ppo: 6, dpo: 3, a2c: 2 },
    { name: 'Mar', ppo: 8, dpo: 4, a2c: 3 },
    { name: 'Apr', ppo: 10, dpo: 5, a2c: 4 },
    { name: 'May', ppo: 12, dpo: 6, a2c: 5 },
    { name: 'Jun', ppo: 14, dpo: 7, a2c: 6 },
  ];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-500">
          Overview of your RLHF training platform
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
        <StatsCard
          title="Total Models"
          value={stats?.totalModels || 0}
          change={12}
          icon={CubeIcon}
          color="text-blue-600"
        />
        <StatsCard
          title="Active Training"
          value={stats?.activeTraining || 0}
          change={-5}
          icon={CogIcon}
          color="text-yellow-600"
        />
        <StatsCard
          title="Total Datasets"
          value={stats?.totalDatasets || 0}
          change={8}
          icon={DocumentTextIcon}
          color="text-green-600"
        />
        <StatsCard
          title="Evaluations"
          value={stats?.totalEvaluations || 0}
          change={15}
          icon={ChartBarIcon}
          color="text-purple-600"
        />
        <StatsCard
          title="Inferences"
          value={stats?.totalInferences || 0}
          change={23}
          icon={PlayIcon}
          color="text-indigo-600"
        />
        <StatsCard
          title="Feedback Collected"
          value={stats?.feedbackCollected || 0}
          change={7}
          icon={DocumentTextIcon}
          color="text-pink-600"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Training Progress Chart */}
        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Training Progress</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trainingData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="ppo" stroke="#3b82f6" strokeWidth={2} />
              <Line type="monotone" dataKey="dpo" stroke="#10b981" strokeWidth={2} />
              <Line type="monotone" dataKey="a2c" stroke="#f59e0b" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Recent Activity */}
        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Activity</h3>
          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <div className="flex-shrink-0">
                <div className="h-8 w-8 bg-green-100 rounded-full flex items-center justify-center">
                  <CogIcon className="h-4 w-4 text-green-600" />
                </div>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900">Training completed</p>
                <p className="text-sm text-gray-500">PPO model "gpt2-finetuned" finished training</p>
              </div>
              <div className="text-sm text-gray-500">2 hours ago</div>
            </div>

            <div className="flex items-center space-x-3">
              <div className="flex-shrink-0">
                <div className="h-8 w-8 bg-blue-100 rounded-full flex items-center justify-center">
                  <CubeIcon className="h-4 w-4 text-blue-600" />
                </div>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900">Model uploaded</p>
                <p className="text-sm text-gray-500">New DPO model "llama-7b-dpo" uploaded</p>
              </div>
              <div className="text-sm text-gray-500">4 hours ago</div>
            </div>

            <div className="flex items-center space-x-3">
              <div className="flex-shrink-0">
                <div className="h-8 w-8 bg-yellow-100 rounded-full flex items-center justify-center">
                  <DocumentTextIcon className="h-4 w-4 text-yellow-600" />
                </div>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900">Dataset created</p>
                <p className="text-sm text-gray-500">New preference dataset "human-feedback-v2" created</p>
              </div>
              <div className="text-sm text-gray-500">6 hours ago</div>
            </div>

            <div className="flex items-center space-x-3">
              <div className="flex-shrink-0">
                <div className="h-8 w-8 bg-purple-100 rounded-full flex items-center justify-center">
                  <ChartBarIcon className="h-4 w-4 text-purple-600" />
                </div>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900">Evaluation completed</p>
                <p className="text-sm text-gray-500">Model evaluation for "gpt2-finetuned" completed</p>
              </div>
              <div className="text-sm text-gray-500">1 day ago</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 