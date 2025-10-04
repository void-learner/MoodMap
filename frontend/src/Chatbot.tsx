import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ChatBubble from './components/ChatBubble';
import InputField from './components/InputField';
import EmotionFeedback from './components/EmotionFeedback';

const BASE_URL = 'http://127.0.0.1:8000';

interface Message {
  text: string;
  sender: 'user' | 'bot';
  emotions?: { label: string; probability: number }[];  // From backend
  showFeedback?: boolean;
}

const Chatbot: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      text: "Hi there! I'm Goru, How can I help you?",
      sender: 'bot',
    },
  ]);
  const [input, setInput] = useState<string>('');
  const [isTyping, setIsTyping] = useState<boolean>(false);
  const chatRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMessage: Message = { text: input, sender: 'user', showFeedback: true };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      const res = await axios.post(`${BASE_URL}/analyze_emotion`, { text: input });
      const botMessage: Message = {
        text: res.data.generated_text,
        sender: 'bot',
      };
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1].emotions = res.data.emotions;  // Attach to user message
        return [...updated, botMessage];
      });
    } catch (error) {
      console.error('Error:', error);
      setMessages((prev) => [...prev, { text: 'Error connecting to backend.', sender: 'bot' }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleFeedback = async (isCorrect: boolean, trueLabels?: string[]) => {
    const lastUserIndex = messages.length - 2;  // User message before bot
    const userMsg = messages[lastUserIndex];
    if (!userMsg.emotions) return;

    const feedbackData = {
      text: userMsg.text,
      predicted_labels: userMsg.emotions.map(e => e.label),
      is_correct: isCorrect,
      true_labels: isCorrect ? userMsg.emotions.filter(e => e.probability > 0.5).map(e => e.label) : trueLabels || [],
    };
    await axios.post(`${BASE_URL}/feedback`, feedbackData);

    // Hide feedback
    setMessages((prev) => {
      const updated = [...prev];
      updated[lastUserIndex].showFeedback = false;
      return updated;
    });
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden p-4">
      <div ref={chatRef} className="flex-1 overflow-y-auto space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className="relative">
            <ChatBubble message={msg} />
            {msg.sender === 'user' && msg.emotions && (
              <EmotionFeedback
                emotions={msg.emotions}
                showFeedback={msg.showFeedback ?? false}
                onSubmit={handleFeedback}
              />
            )}
            <div className="text-xs text-gray-400 mt-1 ${msg.sender === 'user' ? 'text-right' : 'text-left'}">
              {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: true })}
            </div>
          </div>
        ))}
        {isTyping && <div className="text-gray-500">Bot is typing...</div>}
      </div>
      <InputField input={input} setInput={setInput} onSend={sendMessage} />
    </div>
  );
};

export default Chatbot;