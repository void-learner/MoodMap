import React from 'react';
import { User, Bot} from 'lucide-react';

// define the shape of the props
interface ChatBubbleProps {
    message: { text: string; sender: 'user' | 'bot' };
}

const ChatBubble: React.FC<ChatBubbleProps> = ({message}) => {
    const isUser = message.sender === 'user';
    return (
        <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} items-start space-x-2`}>
            {!isUser && <Bot className="w-6 h-6" />}
            <div className={`max-w-xs px-4 py-2 rounded-lg ${isUser ? 'bg-indigo-300 text-white' : 'bg-gray-200 text-gray-800'}`}>
                {message.text}
            </div>
            {isUser && <User className="w-6 h-6" />}
        </div>
    );
};

export default ChatBubble;