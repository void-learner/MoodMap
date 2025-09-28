// src/components/EmotionFeedback.tsx (updated for popup on No)

import React, { useState } from 'react';
import { Check, X } from 'lucide-react';

interface EmotionFeedbackProps {
  emotions: { label: string; probability: number }[];
  showFeedback: boolean;
  onSubmit: (isCorrect: boolean, trueLabels?: string[]) => Promise<void>;
}

const EmotionFeedback: React.FC<EmotionFeedbackProps> = ({ emotions, showFeedback, onSubmit }) => {
  const [showPopup, setShowPopup] = useState(false);
  const [selectedEmotions, setSelectedEmotions] = useState<Set<string>>(new Set());

  if (!showFeedback) return null;

  // Get top emotion for display (highest prob)
  const topEmotion = emotions[0] || { label: 'Neutral', probability: 0.3 };  // Fallback
  const topLabel = topEmotion.label;
  const topProb = Math.round(topEmotion.probability * 100);

  // Filter emotions > 0.5 for suggestions, sorted by prob desc
  const suggestedEmotions = emotions
    .filter(e => e.probability > 0.5)
    .sort((a, b) => b.probability - a.probability);

  const toggleSelect = (label: string) => {
    const newSelected = new Set(selectedEmotions);
    if (newSelected.has(label)) {
      newSelected.delete(label);
    } else {
      newSelected.add(label);
    }
    setSelectedEmotions(newSelected);
  };

  const handleNoSubmit = () => {
    onSubmit(false, Array.from(selectedEmotions));
    setShowPopup(false);
    setSelectedEmotions(new Set());
  };

  return (
    <div className="flex justify-end items-center space-x-2 mt-1 text-sm">
      <span className="bg-gray-200 px-2 py-1 rounded-full text-gray-700">
        {topLabel} {topProb}%
      </span>
      <span className="text-gray-500">Was this emotion correct?</span>
      <button 
        onClick={() => onSubmit(true)} 
        className="flex items-center text-green-500 hover:text-green-700"
      >
        <Check className="w-4 h-4 mr-1" /> Yes
      </button>
      <button 
        onClick={() => setShowPopup(true)} 
        className="flex items-center text-red-500 hover:text-red-700"
      >
        <X className="w-4 h-4 mr-1" /> No
      </button>

      {showPopup && (
        <div className="absolute right-0 mt-2 w-64 bg-white border rounded-lg shadow-lg p-4 z-10">
          <h3 className="text-sm font-semibold mb-2">Select correct emotions:</h3>
          {suggestedEmotions.length > 0 ? (
            <div className="space-y-2">
              {suggestedEmotions.map((e) => (
                <label key={e.label} className="flex items-center">
                  <input 
                    type="checkbox" 
                    checked={selectedEmotions.has(e.label)}
                    onChange={() => toggleSelect(e.label)}
                    className="mr-2"
                  />
                  {e.label} ({Math.round(e.probability * 100)}%)
                </label>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-500">No suggestions above 50% confidence.</p>
          )}
          <button 
            onClick={handleNoSubmit} 
            disabled={selectedEmotions.size === 0}
            className="mt-4 bg-blue-500 text-white px-4 py-1 rounded disabled:opacity-50"
          >
            Submit
          </button>
          <button 
            onClick={() => setShowPopup(false)} 
            className="ml-2 text-gray-500"
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  );
};

export default EmotionFeedback;