import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ChatBubble from './components/ChatBubble';
import InputField from './components/InputField';
import EmotionFeedback from './components/EmotionFeedback';

const BASE_URL = 'http://127.0.0.1:8000';

interface Message {
  text: string;
  sender: 'user' | 'bot';
  emotions?: { label: string; probability: number }[];
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

  // Auto-scroll
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages]);

  // Load session id if exists
  useEffect(() => {
    const saved = localStorage.getItem('session_id');
    if (saved) console.log('Session resumed:', saved);
  }, []);

  // Debug logs
  useEffect(() => {
    console.log('Current messages:', messages);
  }, [messages]);


  // ⬇️ ---- UPDATED sendMessage() (your new version fully merged) ---- ⬇️
  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      text: input,
      sender: 'user',
      showFeedback: true,
    };

    // Add User Message
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      const sessionId = localStorage.getItem('session_id') || undefined;

      const res = await axios.post(`${BASE_URL}/analyze_emotion`, {
        text: userMessage.text,
        session_id: sessionId,
      });

      if (res.data.session_id) {
        localStorage.setItem('session_id', res.data.session_id);
      }

      const botReply = res.data.generated_text?.trim();

      if (!botReply) {
        console.error('EMPTY BOT REPLY:', res.data);
        setMessages(prev => [...prev, { text: 'Goru is thinking...', sender: 'bot' }]);
        return;
      }

      console.log('Goru replied:', botReply);

      const botMessage: Message = {
        text: botReply,
        sender: 'bot',
      };

      // Attach emotions & add bot reply
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1].emotions = res.data.emotions || [];
        return [...updated, botMessage];
      });

    } catch (error: any) {
      console.error('Backend Error:', error.message || error);
      setMessages(prev => [
        ...prev,
        { text: 'Goru is offline. Check backend terminal.', sender: 'bot' },
      ]);
    } finally {
      setIsTyping(false);
    }
  };
  // ⬆️ ---- END of UPDATED sendMessage ---- ⬆️



  // Feedback Handler
  const handleFeedback = async (isCorrect: boolean, trueLabels?: string[]) => {
    const lastUserIndex = messages.length - 2;
    const userMsg = messages[lastUserIndex];
    if (!userMsg?.emotions) return;

    try {
      await axios.post(`${BASE_URL}/feedback`, {
        text: userMsg.text,
        predicted_labels: userMsg.emotions.map(e => e.label),
        is_correct: isCorrect,
        true_labels: trueLabels || [],
      });
    } catch {}

    setMessages(prev => {
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

            {msg.sender === 'user' && msg.emotions && msg.showFeedback && (
              <EmotionFeedback
                emotions={msg.emotions}
                showFeedback={true}
                onSubmit={handleFeedback}
              />
            )}

            <div
              className={`text-xs text-gray-400 mt-1 ${
                msg.sender === 'user' ? 'text-right' : 'text-left'
              }`}
            >
              {new Date().toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit',
              })}
            </div>
          </div>
        ))}

        {isTyping && (
          <div className="text-gray-500 italic">Goru is typing...</div>
        )}
      </div>

      {/* ENTER TO SEND now works because InputField supports onSend */}
      <InputField input={input} setInput={setInput} onSend={sendMessage} />
    </div>
  );
};

export default Chatbot;
