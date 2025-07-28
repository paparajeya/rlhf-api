import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { toast } from 'react-toastify';
import { 
  PlayIcon, 
  ChartBarIcon, 
  DocumentTextIcon,
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon
} from '@heroicons/react/24/outline';

interface Evaluation {
  id: number;
  model_id: number;
  model_name: string;
  dataset_id: number;
  dataset_name: string;
  metrics: {
    bleu_score?: number;
    rouge_score?: number;
    human_score?: number;
    accuracy?: number;
    [key: string]: any;
  };
  status: 'pending' | 'running' | 'completed' | 'failed';
  created_at: string;
  completed_at?: string;
}

interface Model {
  id: number;
  name: string;
  model_type: string;
  algorithm: string;
}

interface Dataset {
  id: number;
  name: string;
  num_samples: number;
}

const Evaluation: React.FC = () => {
  const [showNewEvaluationModal, setShowNewEvaluationModal] = useState(false);
  const [selectedEvaluation, setSelectedEvaluation] = useState<Evaluation | null>(null);
  const queryClient = useQueryClient();

  // Fetch evaluations
  const { data: evaluations, isLoading } = useQuery<Evaluation[]>(
    'evaluations',
    async () => {
      const response = await fetch('/api/v1/evaluations');
      if (!response.ok) throw new Error('Failed to fetch evaluations');
      return response.json();
    }
  );

  // Fetch models
  const { data: models } = useQuery<Model[]>(
    'models',
    async () => {
      const response = await fetch('/api/v1/models');
      if (!response.ok) throw new Error('Failed to fetch models');
      return response.json();
    }
  );

  // Fetch datasets
  const { data: datasets } = useQuery<Dataset[]>(
    'datasets',
    async () => {
      const response = await fetch('/api/v1/datasets');
      if (!response.ok) throw new Error('Failed to fetch datasets');
      return response.json();
    }
  );

  // Create evaluation mutation
  const createEvaluationMutation = useMutation(
    async (evaluationData: { model_id: number; dataset_id: number }) => {
      const response = await fetch('/api/v1/evaluations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(evaluationData),
      });
      if (!response.ok) throw new Error('Failed to create evaluation');
      return response.json();
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('evaluations');
        setShowNewEvaluationModal(false);
        toast.success('Evaluation created successfully');
      },
      onError: () => {
        toast.error('Failed to create evaluation');
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
        return <ClockIcon className="h-5 w-5 text-gray-600" />;
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

  const handleCreateEvaluation = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const evaluationData = {
      model_id: parseInt(formData.get('model_id') as string),
      dataset_id: parseInt(formData.get('dataset_id') as string),
    };
    createEvaluationMutation.mutate(evaluationData);
  };

  const handleViewMetrics = (evaluation: Evaluation) => {
    setSelectedEvaluation(evaluation);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Model Evaluation</h1>
          <p className="text-gray-600">Evaluate model performance on datasets</p>
        </div>
        <button
          onClick={() => setShowNewEvaluationModal(true)}
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
        >
          <PlayIcon className="h-5 w-5 mr-2" />
          New Evaluation
        </button>
      </div>

      {/* Evaluations List */}
      {isLoading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
        </div>
      ) : (
        <div className="bg-white shadow overflow-hidden sm:rounded-md">
          <ul className="divide-y divide-gray-200">
            {evaluations?.map((evaluation) => (
              <li key={evaluation.id} className="px-6 py-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      {getStatusIcon(evaluation.status)}
                    </div>
                    <div className="ml-4 flex-1">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-gray-900">
                            {evaluation.model_name} on {evaluation.dataset_name}
                          </p>
                          <p className="text-sm text-gray-500">
                            Created: {new Date(evaluation.created_at).toLocaleDateString()}
                          </p>
                        </div>
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(evaluation.status)}`}>
                          {evaluation.status}
                        </span>
                      </div>
                      {evaluation.status === 'completed' && evaluation.metrics && (
                        <div className="mt-2 flex space-x-4 text-sm text-gray-600">
                          {evaluation.metrics.bleu_score && (
                            <span>BLEU: {evaluation.metrics.bleu_score.toFixed(3)}</span>
                          )}
                          {evaluation.metrics.rouge_score && (
                            <span>ROUGE: {evaluation.metrics.rouge_score.toFixed(3)}</span>
                          )}
                          {evaluation.metrics.human_score && (
                            <span>Human: {evaluation.metrics.human_score.toFixed(3)}</span>
                          )}
                          {evaluation.metrics.accuracy && (
                            <span>Accuracy: {evaluation.metrics.accuracy.toFixed(3)}</span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => handleViewMetrics(evaluation)}
                      className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                    >
                      <ChartBarIcon className="h-4 w-4 mr-1" />
                      Metrics
                    </button>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Metrics Modal */}
      {selectedEvaluation && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-20 mx-auto p-5 border w-3/4 max-w-4xl shadow-lg rounded-md bg-white">
            <div className="mt-3">
              <h3 className="text-lg font-medium text-gray-900 mb-4">
                Evaluation Metrics - {selectedEvaluation.model_name} on {selectedEvaluation.dataset_name}
              </h3>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Model</label>
                    <p className="mt-1 text-sm text-gray-900">{selectedEvaluation.model_name}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Dataset</label>
                    <p className="mt-1 text-sm text-gray-900">{selectedEvaluation.dataset_name}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Status</label>
                    <p className="mt-1 text-sm text-gray-900">{selectedEvaluation.status}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Created</label>
                    <p className="mt-1 text-sm text-gray-900">
                      {new Date(selectedEvaluation.created_at).toLocaleString()}
                    </p>
                  </div>
                </div>
                {selectedEvaluation.metrics && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Metrics</label>
                    <div className="mt-1 grid grid-cols-2 md:grid-cols-4 gap-4">
                      {Object.entries(selectedEvaluation.metrics).map(([key, value]) => (
                        <div key={key} className="bg-gray-50 p-3 rounded-md">
                          <div className="text-xs font-medium text-gray-500 uppercase">{key}</div>
                          <div className="text-lg font-semibold text-gray-900">
                            {typeof value === 'number' ? value.toFixed(3) : String(value)}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
              <div className="mt-4">
                <button
                  onClick={() => setSelectedEvaluation(null)}
                  className="w-full inline-flex justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* New Evaluation Modal */}
      {showNewEvaluationModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
            <div className="mt-3">
              <h3 className="text-lg font-medium text-gray-900 mb-4">
                Create New Evaluation
              </h3>
              <form onSubmit={handleCreateEvaluation} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Model
                  </label>
                  <select
                    name="model_id"
                    required
                    className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                  >
                    <option value="">Select a model</option>
                    {models?.map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.name} ({model.algorithm})
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Dataset
                  </label>
                  <select
                    name="dataset_id"
                    required
                    className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                  >
                    <option value="">Select a dataset</option>
                    {datasets?.map((dataset) => (
                      <option key={dataset.id} value={dataset.id}>
                        {dataset.name} ({dataset.num_samples} samples)
                      </option>
                    ))}
                  </select>
                </div>
                <div className="flex space-x-3">
                  <button
                    type="submit"
                    className="flex-1 inline-flex justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                  >
                    Create Evaluation
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowNewEvaluationModal(false)}
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

export default Evaluation; 