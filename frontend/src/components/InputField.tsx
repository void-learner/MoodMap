import React from 'react';
import { Send } from 'lucide-react';

interface InputFieldProps {
    input: string;
    setInput: React.Dispatch<React.SetStateAction<string>>;
    onSend: () => void;
}

const InputField: React.FC<InputFieldProps> = ({ input, setInput, onSend }) => {
    return (
        <div className="flex items-center border-t p-4 bg-white">
            <input
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                placeholder='Type your message...'
                className='flex-1 px-4 py-2 border rounded-l-lg focus:outline-none'/>
                <button onClick={onSend} className='bg-indigo-300 text-white px-4 py-2.5 rounded-r-lg hover:bg-indigo-400'>
                    <Send className='w-5 h-5' />
                </button>
        </div>
    );
};

export default InputField;
