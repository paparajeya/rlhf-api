import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { toast } from 'react-toastify';
import { 
  ThumbUpIcon, 
  ThumbDownIcon, 
  ArrowRightIcon,
  CheckCircleIcon,
  XCircleIcon
} from '@heroicons/react/24/outline';

interface FeedbackItem {
  id: number;
  prompt: string;
  response_a: string;
  response_b: string;
  preferred_response?: 'A' | 'B';
  user_id?: string;
  session_id?: string;
  created_at: string;
}

const Feedback: React.FC = () => {
  const [currentItem, setCurrentItem] = useState<FeedbackItem | null>(null);
  const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  const queryClient = useQueryClient();

  // Fetch feedback items
  const { data: feedbackItems, isLoading, refetch } = useQuery<FeedbackItem[]>(
    'feedback-items',
    async () => {
      const response = await fetch('/api/v1/feedback/items');
      if (!response.ok) throw new Error('Failed to fetch feedback items');
      return response.json();
    }
  );

  // Submit feedback mutation
  const submitFeedbackMutation = useMutation(
    async (feedbackData: { item_id: number; preferred_response: 'A' | 'B' }) => {
      const response = await fetch('/api/v1/feedback/submit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...feedbackData,
          session_id: sessionId,
        }),
      });
      if (!response.ok) throw new Error('Failed to submit feedback');
      return response.json();
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('feedback-items');
        toast.success('Feedback submitted successfully');
        getNextItem();
      },
      onError: () => {
        toast.error('Failed to submit feedback');
      },
    }
  );

  // Get next feedback item
  const getNextItem = () => {
    if (feedbackItems && feedbackItems.length > 0) {
      const unratedItem = feedbackItems.find(item => !item.preferred_response);
      if (unratedItem) {
        setCurrentItem(unratedItem);
      } else {
        setCurrentItem(null);
        toast.info('All feedback items have been rated!');
      }
    }
  };

  useEffect(() => {
    if (feedbackItems && feedbackItems.length > 0) {
      getNextItem();
    }
  }, [feedbackItems]);

  const handlePreference = (preferredResponse: 'A' | 'B') => {
    if (currentItem) {
      submitFeedbackMutation.mutate({
        item_id: currentItem.id,
        preferred_response: preferredResponse,
      });
    }
  };

  const handleSkip = () => {
    getNextItem();
  };

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  if (!currentItem) {
    return (
      <div className="space-y-6">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">Human Feedback Collection</h1>
          <p className="text-gray-600 mb-8">
            Help improve our models by providing feedback on generated responses.
          </p>
          
          {feedbackItems && feedbackItems.length === 0 ? (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
              <CheckCircleIcon className="h-12 w-12 text-green-600 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Feedback Items Available</h3>
              <p className="text-gray-600">
                All feedback items have been completed. Check back later for new items.
              </p>
            </div>
          ) : (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
              <XCircleIcon className="h-12 w-12 text-yellow-600 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No More Items to Rate</h3>
              <p className="text-gray-600">
                You've completed all available feedback items. Thank you for your contributions!
              </p>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-2xl font-bold text-gray-900">Human Feedback Collection</h1>
        <p className="text-gray-600">
          Help improve our models by providing feedback on generated responses.
        </p>
      </div>

      {/* Progress */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-700">Progress</span>
          <span className="text-sm text-gray-500">
            {feedbackItems?.filter(item => item.preferred_response).length || 0} / {feedbackItems?.length || 0}
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
            style={{ 
              width: `${feedbackItems ? (feedbackItems.filter(item => item.preferred_response).length / feedbackItems.length) * 100 : 0}%` 
            }}
          ></div>
        </div>
      </div>

      {/* Feedback Item */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="mb-6">
          <h3 className="text-lg font-medium text-gray-900 mb-2">Prompt</h3>
          <p className="text-gray-700 bg-gray-50 p-4 rounded-md">
            {currentItem.prompt}
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className="border-2 border-gray-200 rounded-lg p-4 hover:border-indigo-300 transition-colors">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Response A</h4>
            <p className="text-gray-700 mb-4">{currentItem.response_a}</p>
            <button
              onClick={() => handlePreference('A')}
              className="w-full inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              <ThumbUpIcon className="h-4 w-4 mr-2" />
              Prefer A
            </button>
          </div>

          <div className="border-2 border-gray-200 rounded-lg p-4 hover:border-indigo-300 transition-colors">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Response B</h4>
            <p className="text-gray-700 mb-4">{currentItem.response_b}</p>
            <button
              onClick={() => handlePreference('B')}
              className="w-full inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              <ThumbUpIcon className="h-4 w-4 mr-2" />
              Prefer B
            </button>
          </div>
        </div>

        <div className="flex justify-center">
          <button
            onClick={handleSkip}
            className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            <ArrowRightIcon className="h-4 w-4 mr-2" />
            Skip
          </button>
        </div>
      </div>

      {/* Instructions */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h3 className="text-sm font-medium text-blue-900 mb-2">Instructions</h3>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• Read the prompt carefully</li>
          <li>• Compare the two responses (A and B)</li>
          <li>• Choose the response you prefer based on helpfulness, accuracy, and safety</li>
          <li>• If both responses are equally good or bad, you can skip</li>
          <li>• Your feedback helps improve our models</li>
        </ul>
      </div>
    </div>
  );
};

export default Feedback; 