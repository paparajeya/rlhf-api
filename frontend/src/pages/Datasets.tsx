import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { toast } from 'react-toastify';
import { 
  PlusIcon, 
  TrashIcon, 
  EyeIcon, 
  DocumentTextIcon,
  CloudArrowUpIcon,
  DocumentIcon
} from '@heroicons/react/24/outline';

interface Dataset {
  id: number;
  name: string;
  description: string;
  file_size: number;
  num_samples: number;
  created_at: string;
  metadata?: any;
}

const Datasets: React.FC = () => {
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const queryClient = useQueryClient();

  // Fetch datasets
  const { data: datasets, isLoading } = useQuery<Dataset[]>(
    'datasets',
    async () => {
      const response = await fetch('/api/v1/datasets');
      if (!response.ok) throw new Error('Failed to fetch datasets');
      return response.json();
    }
  );

  // Delete dataset mutation
  const deleteDatasetMutation = useMutation(
    async (datasetId: number) => {
      const response = await fetch(`/api/v1/datasets/${datasetId}`, {
        method: 'DELETE',
      });
      if (!response.ok) throw new Error('Failed to delete dataset');
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('datasets');
        toast.success('Dataset deleted successfully');
      },
      onError: () => {
        toast.error('Failed to delete dataset');
      },
    }
  );

  // Upload dataset mutation
  const uploadDatasetMutation = useMutation(
    async (formData: FormData) => {
      const response = await fetch('/api/v1/datasets/upload', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Failed to upload dataset');
      return response.json();
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('datasets');
        setShowUploadModal(false);
        toast.success('Dataset uploaded successfully');
      },
      onError: () => {
        toast.error('Failed to upload dataset');
      },
    }
  );

  const handleDeleteDataset = (datasetId: number) => {
    if (window.confirm('Are you sure you want to delete this dataset?')) {
      deleteDatasetMutation.mutate(datasetId);
    }
  };

  const handleViewDataset = (dataset: Dataset) => {
    setSelectedDataset(dataset);
  };

  const handleUploadDataset = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    uploadDatasetMutation.mutate(formData);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Datasets</h1>
          <p className="text-gray-600">Manage your training datasets</p>
        </div>
        <button
          onClick={() => setShowUploadModal(true)}
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
        >
          <PlusIcon className="h-5 w-5 mr-2" />
          Upload Dataset
        </button>
      </div>

      {/* Datasets Grid */}
      {isLoading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {datasets?.map((dataset) => (
            <div
              key={dataset.id}
              className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow"
            >
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">{dataset.name}</h3>
                  <p className="text-sm text-gray-600">{dataset.description}</p>
                </div>
                <DocumentIcon className="h-8 w-8 text-indigo-600" />
              </div>

              <div className="space-y-2 mb-4">
                <div className="flex items-center text-sm text-gray-600">
                  <DocumentTextIcon className="h-4 w-4 mr-2" />
                  Samples: {dataset.num_samples.toLocaleString()}
                </div>
                <div className="flex items-center text-sm text-gray-600">
                  <CloudArrowUpIcon className="h-4 w-4 mr-2" />
                  Size: {formatFileSize(dataset.file_size)}
                </div>
                <div className="text-sm text-gray-500">
                  Created: {new Date(dataset.created_at).toLocaleDateString()}
                </div>
              </div>

              <div className="flex space-x-2">
                <button
                  onClick={() => handleViewDataset(dataset)}
                  className="flex-1 inline-flex items-center justify-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                >
                  <EyeIcon className="h-4 w-4 mr-1" />
                  View
                </button>
                <button
                  onClick={() => handleDeleteDataset(dataset.id)}
                  className="inline-flex items-center justify-center px-3 py-2 border border-red-300 shadow-sm text-sm font-medium rounded-md text-red-700 bg-white hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                >
                  <TrashIcon className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Dataset Details Modal */}
      {selectedDataset && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-20 mx-auto p-5 border w-3/4 max-w-4xl shadow-lg rounded-md bg-white">
            <div className="mt-3">
              <h3 className="text-lg font-medium text-gray-900 mb-4">
                Dataset Details - {selectedDataset.name}
              </h3>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Name</label>
                    <p className="mt-1 text-sm text-gray-900">{selectedDataset.name}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Description</label>
                    <p className="mt-1 text-sm text-gray-900">{selectedDataset.description}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">File Size</label>
                    <p className="mt-1 text-sm text-gray-900">{formatFileSize(selectedDataset.file_size)}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Number of Samples</label>
                    <p className="mt-1 text-sm text-gray-900">{selectedDataset.num_samples.toLocaleString()}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Created</label>
                    <p className="mt-1 text-sm text-gray-900">
                      {new Date(selectedDataset.created_at).toLocaleString()}
                    </p>
                  </div>
                </div>
                {selectedDataset.metadata && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Metadata</label>
                    <pre className="mt-1 text-sm text-gray-900 bg-gray-50 p-2 rounded">
                      {JSON.stringify(selectedDataset.metadata, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
              <div className="mt-4">
                <button
                  onClick={() => setSelectedDataset(null)}
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
                Upload Dataset
              </h3>
              <form onSubmit={handleUploadDataset} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Dataset Name
                  </label>
                  <input
                    type="text"
                    name="name"
                    required
                    className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                    placeholder="Enter dataset name"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Description
                  </label>
                  <textarea
                    name="description"
                    className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                    rows={3}
                    placeholder="Enter dataset description"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Dataset File
                  </label>
                  <input
                    type="file"
                    name="file"
                    required
                    className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                    accept=".json,.csv,.txt"
                  />
                  <p className="mt-1 text-sm text-gray-500">
                    Supported formats: JSON, CSV, TXT
                  </p>
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

export default Datasets; 