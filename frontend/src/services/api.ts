import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create axios instance
export const api = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API endpoints
export const endpoints = {
  // Models
  models: {
    list: () => api.get('/models'),
    get: (id: number) => api.get(`/models/${id}`),
    create: (data: any) => api.post('/models', data),
    update: (id: number, data: any) => api.put(`/models/${id}`, data),
    delete: (id: number) => api.delete(`/models/${id}`),
    upload: (file: File, metadata: any) => {
      const formData = new FormData();
      formData.append('file', file);
      Object.keys(metadata).forEach(key => {
        formData.append(key, metadata[key]);
      });
      return api.post('/models/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
    },
    download: (id: number) => api.get(`/models/${id}/download`),
  },

  // Training
  training: {
    list: () => api.get('/training'),
    get: (id: number) => api.get(`/training/${id}`),
    create: (data: any) => api.post('/training', data),
    start: (id: number) => api.post(`/training/${id}/start`),
    stop: (id: number) => api.post(`/training/${id}/stop`),
    logs: (id: number) => api.get(`/training/${id}/logs`),
  },

  // Datasets
  datasets: {
    list: () => api.get('/datasets'),
    get: (id: number) => api.get(`/datasets/${id}`),
    create: (data: any) => api.post('/datasets', data),
    update: (id: number, data: any) => api.put(`/datasets/${id}`, data),
    delete: (id: number) => api.delete(`/datasets/${id}`),
    upload: (file: File, metadata: any) => {
      const formData = new FormData();
      formData.append('file', file);
      Object.keys(metadata).forEach(key => {
        formData.append(key, metadata[key]);
      });
      return api.post('/datasets/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
    },
  },

  // Feedback
  feedback: {
    list: () => api.get('/feedback'),
    get: (id: number) => api.get(`/feedback/${id}`),
    create: (data: any) => api.post('/feedback', data),
    update: (id: number, data: any) => api.put(`/feedback/${id}`, data),
    delete: (id: number) => api.delete(`/feedback/${id}`),
  },

  // Evaluation
  evaluation: {
    list: () => api.get('/evaluation'),
    get: (id: number) => api.get(`/evaluation/${id}`),
    create: (data: any) => api.post('/evaluation', data),
    run: (modelId: number, datasetId: number) => 
      api.post('/evaluation/run', { model_id: modelId, dataset_id: datasetId }),
  },

  // Inference
  inference: {
    generate: (data: any) => api.post('/inference/generate', data),
    batch: (data: any) => api.post('/inference/batch', data),
  },

  // Health check
  health: () => api.get('/health'),
};

// Types
export interface Model {
  id: number;
  name: string;
  description?: string;
  model_path: string;
  model_type: string;
  algorithm: string;
  config?: any;
  metrics?: any;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface TrainingJob {
  id: number;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  algorithm: string;
  config: any;
  progress: number;
  metrics?: any;
  logs?: string;
  error_message?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

export interface Dataset {
  id: number;
  name: string;
  description?: string;
  file_path: string;
  file_size: number;
  num_samples: number;
  metadata?: any;
  created_at: string;
}

export interface Feedback {
  id: number;
  prompt: string;
  preferred_response: string;
  dispreferred_response: string;
  user_id?: string;
  session_id?: string;
  created_at: string;
}

export interface Evaluation {
  id: number;
  model_id: number;
  dataset_id: number;
  metrics: any;
  created_at: string;
}

// Error handling
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: any
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

// Utility functions
export const handleApiError = (error: any): ApiError => {
  if (error.response) {
    return new ApiError(
      error.response.data?.detail || error.response.data?.message || 'API Error',
      error.response.status,
      error.response.data
    );
  }
  return new ApiError(error.message || 'Network Error', 0);
}; 