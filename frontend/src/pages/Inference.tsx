import React, { useState } from 'react';
import { useQuery, useMutation } from 'react-query';
import { toast } from 'react-toastify';
import { 
  PlayIcon, 
  DocumentTextIcon,
  SparklesIcon,
  ClipboardIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';

interface Model {
  id: number;
  name: string;
  model_type: string;
  algorithm: string;
  is_active: boolean;
}

interface InferenceResponse {
  response: string;
  model_name: string;
  inference_time: number;
  tokens_generated: number;
}

const Inference: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [selectedModel, setSelectedModel] = useState<number | null>(null);
  const [maxLength, setMaxLength] = useState(100);
  const [temperature, setTemperature] = useState(0.7);
  const [responses, setResponses] = useState<InferenceResponse[]>([]);

  // Fetch models
  const { data: models, isLoading } = useQuery<Model[]>(
    'models',
    async () => {
      const response = await fetch('/api/v1/models');
      if (!response.ok) throw new Error('Failed to fetch models');
      return response.json();
    }
  );

  // Run inference mutation
  const inferenceMutation = useMutation(
    async (inferenceData: {
      model_id: number;
      prompt: string;
      max_length: number;
      temperature: number;
    }) => {
      const response = await fetch('/api/v1/inference/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(inferenceData),
      });
      if (!response.ok) throw new Error('Failed to generate response');
      return response.json();
    },
    {
      onSuccess: (data: InferenceResponse) => {
        setResponses(prev => [data, ...prev]);
        toast.success('Response generated successfully');
      },
      onError: () => {
        toast.error('Failed to generate response');
      },
    }
  );

  const handleGenerate = () => {
    if (!prompt.trim()) {
      toast.error('Please enter a prompt');
      return;
    }
    if (!selectedModel) {
      toast.error('Please select a model');
      return;
    }

    inferenceMutation.mutate({
      model_id: selectedModel,
      prompt: prompt.trim(),
      max_length: maxLength,
      temperature: temperature,
    });
  };

  const handleClear = () => {
    setPrompt('');
    setResponses([]);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success('Copied to clipboard');
  };

  const activeModels = models?.filter(model => model.is_active) || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Model Inference</h1>
        <p className="text-gray-600">Generate responses using your trained models</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Panel */}
        <div className="space-y-6">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Generate Response</h3>
            
            {/* Model Selection */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select Model
              </label>
              {isLoading ? (
                <div className="animate-pulse bg-gray-200 h-10 rounded-md"></div>
              ) : (
                <select
                  value={selectedModel || ''}
                  onChange={(e) => setSelectedModel(Number(e.target.value) || null)}
                  className="block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                >
                  <option value="">Select a model</option>
                  {activeModels.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name} ({model.algorithm})
                    </option>
                  ))}
                </select>
              )}
            </div>

            {/* Prompt Input */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Prompt
              </label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={6}
                className="block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                placeholder="Enter your prompt here..."
              />
            </div>

            {/* Parameters */}
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Max Length
                </label>
                <input
                  type="number"
                  value={maxLength}
                  onChange={(e) => setMaxLength(Number(e.target.value))}
                  min="1"
                  max="1000"
                  className="block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Temperature
                </label>
                <input
                  type="number"
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  min="0"
                  max="2"
                  step="0.1"
                  className="block w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                />
              </div>
            </div>

            {/* Actions */}
            <div className="flex space-x-3">
              <button
                onClick={handleGenerate}
                disabled={inferenceMutation.isLoading || !prompt.trim() || !selectedModel}
                className="flex-1 inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {inferenceMutation.isLoading ? (
                  <ArrowPathIcon className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <SparklesIcon className="h-4 w-4 mr-2" />
                )}
                {inferenceMutation.isLoading ? 'Generating...' : 'Generate'}
              </button>
              <button
                onClick={handleClear}
                className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                Clear
              </button>
            </div>
          </div>
        </div>

        {/* Responses Panel */}
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-medium text-gray-900">Generated Responses</h3>
            {responses.length > 0 && (
              <button
                onClick={() => setResponses([])}
                className="text-sm text-gray-500 hover:text-gray-700"
              >
                Clear All
              </button>
            )}
          </div>

          {responses.length === 0 ? (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
              <DocumentTextIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Responses Yet</h3>
              <p className="text-gray-600">
                Generate a response to see it here
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {responses.map((response, index) => (
                <div key={index} className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <h4 className="text-sm font-medium text-gray-900">{response.model_name}</h4>
                      <p className="text-xs text-gray-500">
                        {response.inference_time.toFixed(2)}s • {response.tokens_generated} tokens
                      </p>
                    </div>
                    <button
                      onClick={() => copyToClipboard(response.response)}
                      className="text-gray-400 hover:text-gray-600"
                      title="Copy to clipboard"
                    >
                      <ClipboardIcon className="h-4 w-4" />
                    </button>
                  </div>
                  <div className="bg-gray-50 rounded-md p-3">
                    <p className="text-sm text-gray-700 whitespace-pre-wrap">
                      {response.response}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Instructions */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="text-sm font-medium text-blue-900 mb-2">Instructions</h3>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• Select an active model from the dropdown</li>
          <li>• Enter your prompt in the text area</li>
          <li>• Adjust max length and temperature parameters as needed</li>
          <li>• Click "Generate" to create a response</li>
          <li>• Use the copy button to copy responses to clipboard</li>
        </ul>
      </div>
    </div>
  );
};

export default Inference; 