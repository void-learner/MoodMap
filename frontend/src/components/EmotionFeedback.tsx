import React, { useState } from 'react';
import { Check, X } from 'lucide-react';

interface EmotionFeedbackProps {
    emotions: {label: string, probability: number}[];
    showFeedback: boolean;
    onSubmit: (isCorrect: boolean, trueLabels?: string[]) => Promise<void>;
}

const EmotionFeedback: React.FC<EmotionFeedbackProps> = ({emotions, showFeedback, onSubmit }) => {
    const [correctLables, setCorrectLables] = useState<string>('');

    if (!showFeedback) return null;

    const handleNo = () => {
        const labels = correctLables.split(',').map(label => label.trim());
        onSubmit(false, labels);
    };

    return (
        <div className='flex justify-end items-center space-x-2 mt-1'>
            <span className='text-sm text-gray-600'>
                {emotions.map(e => `${e.label} ${Math.round(e.probability * 100)}%`).join(', ')}
            </span>
            <button onClick={() => onSubmit(true)} className='text-green-600'><Check className='w-4 h-4' /></button>
            <button onClick={handleNo} className='text-red-700'><X className='w-4 h-4' /></button>
            {!emotions.every(e => e.probability > 0.5) && (
                <input
                    type='text'
                    value={correctLables}
                    onChange={e => setCorrectLables(e.target.value)}
                    placeholder='Correct labels'
                    className="text-sm border p-1 rounded"
                />
            )}
        </div>
    );
};

export default EmotionFeedback;
