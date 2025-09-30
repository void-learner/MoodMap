import React, { useEffect, useState, useRef } from 'react';
import axios from 'axios'; // Import axios for making HTTP requests
import ChatBubble from './components/ChatBubble';
import ChatInput from './components/InputField';
import EmotionFeedback from './components/EmotionFeedback';

interface Message {
    text: string;
    sender: 'user' | 'bot';
    emotion? : {label: string, probability: number};  // optional
    showFeedback?: boolean;  // optional 
}

const [message, setMessage] = useState<Message[]>([
    { text: "Hi there! ...I am goru", sender: "bot"},
]);

const [input, setInput] = useState<string>('');
const [isLoading, setIsTyping] = useState<boolean>(false);
const ChatRef = useRef<HTMLDivElement>(null);

useEffect(() => {
    if (ChatRef.current) {
        ChatRef.current.scrollTop = ChatRef.current.scrollHeight;
    }
},[message]);

const sendMessage = async () => {
    if (!input.trim()) return; // Prevent sending empty messages

    const userMessage: Message = {text: input, sender: 'user', showFeedback: false};
    setMessage((prevMessages) => [...prevMessages, userMessage]);
    setInput('');
    setIsTyping(true);

    try{
        const response = await axios.post('http://localhost:8000/analyze_emotion', { text: input });
        const botMessage: Message = {
            text: response.data.generated_text,
            sender: 'bot',
        };

        setMessage((prevMessages) => {
            const update = [...prevMessages];
            update[update.length - 1].emotion = response.data.emotion;
            return update;
        });
    } catch (error) {
        console.error("Error fetching bot response:", error);
        setMessage((prevMessages) => [...prevMessages, { text: "Sorry, something went wrong.", sender: "bot" }]);
    } finally {
        setIsTyping(false);
    }    
};

const handleFeedback = async (isCorrect: boolean, trueLabel?: string[]) => {
    const lastUserIndex = message.length - 2;
    const userMessage = message[lastUserIndex];
    if (!userMessage.emotion) return;

    const feedbackData = {
        text: userMessage.text,
        predicted_label: userMessage.emotion.label,
        is_correct: isCorrect,
        true_label: isCorrect ? [userMessage.emotion.label] : trueLabel || [],

    }
    await axios.post('http://localhost:8000/feedback', feedbackData);

    // Hide feedback
    
}
