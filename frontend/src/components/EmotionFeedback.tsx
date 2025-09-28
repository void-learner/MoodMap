import React, { useState } from 'react';
import { Check, X } from 'lucide-react';

interface EmotionFeedbackProps {
    emotions: {label: string, probability: number}[];
    showFeedback: boolean;
    onSubmit: (isCorrect: boolean, trueLabels?: string[]) => Promise<void>;
}

const EmotionFeedback: React.FC<EmotionFeedbackProps> = ({emotions, showFeedback, onSubmit }) => {
    const [showPopup, setShowPopup] = useState(false);
    const [selectedEmotions, setSelectedEmotions] = useState<Set<string>>(new Set());

    if (!showFeedback) return null;

    // Get top emotions
    const topEmotions = emotions[0] || {label: 'Neutral', probability: 0.3};
    const topLabel = topEmotions.label;
    const topProbability = Math.round(topEmotions.probability * 100);

    // Filter emotions with probability > 0.5
    const filteremotions = emotions
        .filter(e => e.probability > 0.5)
        .sort((a, b) => b.probability - a.probability);

    // check if label already in the set
    const toggleSelect = (label: string) => {
        const newSelected = new Set(selectedEmotions);
        if (newSelected.has(label)) {
            newSelected.delete(label);
        } else {
            newSelected.add(label);
        }
        setSelectedEmotions(newSelected);
    };

    // Handle no click
    const handleNo = () => {
        onSubmit(false, Array.from(selectedEmotions));
        setShowPopup(false);
        setSelectedEmotions(new Set());
    };


    return (
        <div className='flex justify-end items-center space-x-2 mt-1 text-sm'>
            <span className='bg-gray-200 px-2 py-1 rounded-full text-gray-700'>
                {topLabel} {topProbability}%
            </span>
            <button
                onClick={() => onSubmit(true)}
                className='flex items-center text-green-500 hover:text-green-700'
            >
                <Check className='mr-1' size={16} />
                Yes
            </button>
            <button
                onClick={() => setShowPopup(true)}
                className='flex items-center text-red-500 hover:text-red-700'
            >
                <X className='mr-1' size={16} />
                No
            </button>
        </div>
    );
};

export default EmotionFeedback;
