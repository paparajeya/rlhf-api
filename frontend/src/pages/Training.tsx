import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { toast } from 'react-toastify';
import { 
  PlayIcon, 
  StopIcon, 
  EyeIcon, 
  PlusIcon,
  ClockIcon,
  CheckCircleIcon,
  XCircleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';

interface TrainingJob {
  id: number;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  algorithm: string;
  progress: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
}

const Training: React.FC = () => {
  const [showNewJobModal, setShowNewJobModal] = useState(false);
  const [selectedJob, setSelectedJob] = useState<TrainingJob | null>(null);
  const queryClient = useQueryClient();

  // Fetch training jobs
  const { data: jobs, isLoading } = useQuery<TrainingJob[]>(
    'training-jobs',
    async () => {
      const response = await fetch('/api/v1/training');
      if (!response.ok) throw new Error('Failed to fetch training jobs');
      return response.json();
    },
    {
      refetchInterval: 5000, // Refetch every 5 seconds for real-time updates
    }
  );

  // Start training job mutation
  const startJobMutation = useMutation(
    async (jobId: number) => {
      const response = await fetch(`/api/v1/training/${jobId}/start`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error('Failed to start training job');
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('training-jobs');
        toast.success('Training job started successfully');
      },
      onError: () => {
        toast.error('Failed to start training job');
      },
    }
  );

  // Stop training job mutation
  const stopJobMutation = useMutation(
    async (jobId: number) => {
      const response = await fetch(`/api/v1/training/${jobId}/stop`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error('Failed to stop training job');
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('training-jobs');
        toast.success('Training job stopped successfully');
      },
      onError: () => {
        toast.error('Failed to stop training job');
      },
    }
  );

  // Create new training job mutation
  const createJobMutation = useMutation(
    async (jobData: any) => {
      const response = await fetch('/api/v1/training', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(jobData),
      });
      if (!response.ok) throw new Error('Failed to create training job');
      return response.json();
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('training-jobs');
        setShowNewJobModal(false);
        toast.success('Training job created successfully');
      },
      onError: () => {
        toast.error('Failed to create training job');
      },
    }
  );

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <PlayIcon className="h-5 w-5 text-blue-600" />;
      case 'completed':
        return <CheckCircleIcon className="h-5 w-5 text-green-600" />;
      case 'failed':
        return <XCircleIcon className="h-5 w-5 text-red-600" />;
      case 'pending':
        return <ClockIcon className="h-5 w-5 text-yellow-600" />;
      default:
        return <ExclamationTriangleIcon className="h-5 w-5 text-gray-600" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'bg-blue-100 text-blue-800';
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      case 'pending':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const handleStartJob = (jobId: number) => {
    startJobMutation.mutate(jobId);
  };

  const handleStopJob = (jobId: number) => {
    stopJobMutation.mutate(jobId);
  };

  const handleViewLogs = (job: TrainingJob) => {
    setSelectedJob(job);
  };

  const handleCreateJob = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const jobData = {
      name: formData.get('name') as string,
      algorithm: formData.get('algorithm') as string,
      config: {
        learning_rate: parseFloat(formData.get('learning_rate') as string),
        batch_size: parseInt(formData.get('batch_size') as string),
        epochs: parseInt(formData.get('epochs') as string),
      },
    };
    createJobMutation.mutate(jobData);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Training Jobs</h1>
          <p className="text-gray-600">Manage and monitor your RLHF training jobs</p>
        </div>
        <button
          onClick={() => setShowNewJobModal(true)}
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
        >
          <PlusIcon className="h-5 w-5 mr-2" />
          New Training Job
        </button>
      </div>

      {/* Training Jobs List */}
      {isLoading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
        </div>
      ) : (
        <div className="bg-white shadow overflow-hidden sm:rounded-md">
          <ul className="divide-y divide-gray-200">
            {jobs?.map((job) => (
              <li key={job.id} className="px-6 py-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      {getStatusIcon(job.status)}
                    </div>
                    <div className="ml-4 flex-1">
                      <div className="flex items-center justify-between">
                        <p className="text-sm font-medium text-gray-900">{job.name}</p>
                        <div className="flex items-center space-x-2">
                          <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(job.status)}`}>
                            {job.status}
                          </span>
                          {job.status === 'running' && (
                            <span className="text-sm text-gray-500">
                              {Math.round(job.progress * 100)}%
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="mt-2 flex items-center text-sm text-gray-500">
                        <span className="mr-4">Algorithm: {job.algorithm}</span>
                        <span>Created: {new Date(job.created_at).toLocaleDateString()}</span>
                      </div>
                      {job.status === 'running' && (
                        <div className="mt-2">
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${job.progress * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      )}
                      {job.error_message && (
                        <div className="mt-2 text-sm text-red-600">
                          Error: {job.error_message}
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => handleViewLogs(job)}
                      className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                    >
                      <EyeIcon className="h-4 w-4 mr-1" />
                      Logs
                    </button>
                    {job.status === 'pending' && (
                      <button
                        onClick={() => handleStartJob(job.id)}
                        className="inline-flex items-center px-3 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
                      >
                        <PlayIcon className="h-4 w-4 mr-1" />
                        Start
                      </button>
                    )}
                    {job.status === 'running' && (
                      <button
                        onClick={() => handleStopJob(job.id)}
                        className="inline-flex items-center px-3 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                      >
                        <StopIcon className="h-4 w-4 mr-1" />
                        Stop
                      </button>
                    )}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Logs Modal */}
      {selectedJob && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-20 mx-auto p-5 border w-3/4 max-w-4xl shadow-lg rounded-md bg-white">
            <div className="mt-3">
              <h3 className="text-lg font-medium text-gray-900 mb-4">
                Training Logs - {selectedJob.name}
              </h3>
              <div className="bg-gray-900 text-green-400 p-4 rounded-md font-mono text-sm h-96 overflow-y-auto">
                <pre>
                  {selectedJob.logs || 'No logs available'}
                </pre>
              </div>
              <div className="mt-4">
                <button
                  onClick={() => setSelectedJob(null)}
                  className="w-full inline-flex justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* New Job Modal */}
      {showNewJobModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
            <div className="mt-3">
              <h3 className="text-lg font-medium text-gray-900 mb-4">
                Create New Training Job
              </h3>
              <form onSubmit={handleCreateJob} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Job Name
                  </label>
                  <input
                    type="text"
                    name="name"
                    required
                    className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                    placeholder="Enter job name"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Algorithm
                  </label>
                  <select
                    name="algorithm"
                    required
                    className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                  >
                    <option value="ppo">PPO</option>
                    <option value="dpo">DPO</option>
                    <option value="a2c">A2C</option>
                  </select>
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">
                      Learning Rate
                    </label>
                    <input
                      type="number"
                      name="learning_rate"
                      step="0.000001"
                      defaultValue="0.00001"
                      required
                      className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">
                      Batch Size
                    </label>
                    <input
                      type="number"
                      name="batch_size"
                      defaultValue="4"
                      required
                      className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">
                      Epochs
                    </label>
                    <input
                      type="number"
                      name="epochs"
                      defaultValue="3"
                      required
                      className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                    />
                  </div>
                </div>
                <div className="flex space-x-3">
                  <button
                    type="submit"
                    className="flex-1 inline-flex justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                  >
                    Create Job
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowNewJobModal(false)}
                    className="flex-1 inline-flex justify-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                  >
                    Cancel
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Training; 