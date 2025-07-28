import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { toast } from 'react-toastify';
import { 
  PlusIcon, 
  TrashIcon, 
  EyeIcon, 
  PlayIcon,
  DocumentTextIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';

interface Model {
  id: number;
  name: string;
  description: string;
  model_type: string;
  algorithm: string;
  is_active: boolean;
  created_at: string;
  metrics?: any;
}

const Models: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const queryClient = useQueryClient();

  // Fetch models
  const { data: models, isLoading } = useQuery<Model[]>(
    'models',
    async () => {
      const response = await fetch('/api/v1/models');
      if (!response.ok) throw new Error('Failed to fetch models');
      return response.json();
    }
  );

  // Delete model mutation
  const deleteModelMutation = useMutation(
    async (modelId: number) => {
      const response = await fetch(`/api/v1/models/${modelId}`, {
        method: 'DELETE',
      });
      if (!response.ok) throw new Error('Failed to delete model');
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('models');
        toast.success('Model deleted successfully');
      },
      onError: (error) => {
        toast.error('Failed to delete model');
      },
    }
  );

  const handleDeleteModel = (modelId: number) => {
    if (window.confirm('Are you sure you want to delete this model?')) {
      deleteModelMutation.mutate(modelId);
    }
  };

  const handleViewMetrics = (model: Model) => {
    setSelectedModel(model);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Models</h1>
          <p className="text-gray-600">Manage your trained RLHF models</p>
        </div>
        <button
          onClick={() => setShowUploadModal(true)}
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
        >
          <PlusIcon className="h-5 w-5 mr-2" />
          Upload Model
        </button>
      </div>

      {/* Models Grid */}
      {isLoading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {models?.map((model) => (
            <div
              key={model.id}
              className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow"
            >
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">{model.name}</h3>
                  <p className="text-sm text-gray-600">{model.description}</p>
                </div>
                <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                  model.is_active 
                    ? 'bg-green-100 text-green-800' 
                    : 'bg-gray-100 text-gray-800'
                }`}>
                  {model.is_active ? 'Active' : 'Inactive'}
                </span>
              </div>

              <div className="space-y-2 mb-4">
                <div className="flex items-center text-sm text-gray-600">
                  <DocumentTextIcon className="h-4 w-4 mr-2" />
                  Type: {model.model_type}
                </div>
                <div className="flex items-center text-sm text-gray-600">
                  <PlayIcon className="h-4 w-4 mr-2" />
                  Algorithm: {model.algorithm}
                </div>
                <div className="text-sm text-gray-500">
                  Created: {new Date(model.created_at).toLocaleDateString()}
                </div>
              </div>

              <div className="flex space-x-2">
                <button
                  onClick={() => handleViewMetrics(model)}
                  className="flex-1 inline-flex items-center justify-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                >
                  <ChartBarIcon className="h-4 w-4 mr-1" />
                  Metrics
                </button>
                <button
                  onClick={() => handleDeleteModel(model.id)}
                  className="inline-flex items-center justify-center px-3 py-2 border border-red-300 shadow-sm text-sm font-medium rounded-md text-red-700 bg-white hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                >
                  <TrashIcon className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Metrics Modal */}
      {selectedModel && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
            <div className="mt-3">
              <h3 className="text-lg font-medium text-gray-900 mb-4">
                Metrics for {selectedModel.name}
              </h3>
              {selectedModel.metrics ? (
                <div className="space-y-2">
                  {Object.entries(selectedModel.metrics).map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span className="text-sm font-medium text-gray-700">{key}:</span>
                      <span className="text-sm text-gray-900">{String(value)}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-gray-500">No metrics available</p>
              )}
              <div className="mt-4">
                <button
                  onClick={() => setSelectedModel(null)}
                  className="w-full inline-flex justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
            <div className="mt-3">
              <h3 className="text-lg font-medium text-gray-900 mb-4">
                Upload Model
              </h3>
              <form className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Model Name
                  </label>
                  <input
                    type="text"
                    className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                    placeholder="Enter model name"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Description
                  </label>
                  <textarea
                    className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                    rows={3}
                    placeholder="Enter model description"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Model File
                  </label>
                  <input
                    type="file"
                    className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                    accept=".pt,.pth,.bin"
                  />
                </div>
                <div className="flex space-x-3">
                  <button
                    type="submit"
                    className="flex-1 inline-flex justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                  >
                    Upload
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowUploadModal(false)}
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

export default Models; 