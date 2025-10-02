import React, { useEffect, useState } from "react";
import { Brain } from 'lucide-react';
import Chatbot from './Chatbot';


// Extend the Window interface to include electronAPI
declare global {
  interface Window {
    electronAPI?: {
      // ping: () => void;
    };
  }
}


const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-white shadow-md p-4 flex items-center space-x-2">
        <div className="bg-blue-100 p-2 rounded">
          <Brain className="w-6 h-6 text-blue-500" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-gray-800">BERT-Powered Emotion Chatbot</h1>
          <p className="text-sm text-gray-500">Fine-tuned on GoEmotions dataset with continuous learning</p>
        </div>
        <div className="ml-auto flex space-x-4 text-gray-500">
          <button>Stats</button>  {/* Placeholder; add logic if needed */}
        </div>
      </header>
      <Chatbot />
    </div>
  );
};



// === IGNORE: Below are test snippets for individual components ===

// import ChatBubble from "./components/ChatBubble";
// import InputField from "./components/InputField";
// import EmotionFeedback from "./components/EmotionFeedback";


// function App() {
//   const [feedbackDone, setFeedbackDone] = useState(false);

//   const sampleEmotions = [
//     { label: "Happy", probability: 0.8 },
//     { label: "Surprised", probability: 0.3 },
//   ];

//   const handleFeedback = async (isCorrect: boolean, trueLabels?: string[]) => {
//     console.log("Feedback submitted:", { isCorrect, trueLabels });
//     setFeedbackDone(true);
//   };

//   return (
//     <div className="p-4">
//       <h1>EmotionFeedback Test</h1>
//       {!feedbackDone && (
//         <EmotionFeedback
//           emotions={sampleEmotions}
//           showFeedback={true}
//           onSubmit={handleFeedback}
//         />
//       )}
//     </div>
//   );
// }




// function App() {
//   const [input, setInput] = useState<string>("");

//   const handleSend = () => {
//     console.log("Message sent:", input); 
//     setInput(""); // clear after sending
//   };

//   return (
//     <div className="p-6">
//       <h1 className="text-xl font-bold mb-4">Test InputField</h1>
//       <InputField input={input} setInput={setInput} onSend={handleSend} />
//     </div>
//   );
// }




// function App() {
//   return (
//     <div className="p-4 space-y-4">
//       <ChatBubble message={{ text: "Hello! I'm the bot ", sender: "bot" }} />
//       <ChatBubble message={{ text: "Hi there! I'm the user ", sender: "user" }} />
//       <ChatBubble message={{ text: "How are you doing?", sender: "bot" }} />
//       <ChatBubble message={{ text: "I'm doing great, thanks!", sender: "user" }} />
//     </div>
//   );
// }


// function App() {
//   useEffect(() => {
//     // Function to check if electronAPI is available
//     const checkElectronAPI = () => {
//       if (window.electronAPI) {
//         console.log("electronAPI is available in React");
//         window.electronAPI.ping(); // logs "Ping from preload" in terminal
//       } else {
//         console.log("electronAPI NOT found, retrying...");
//       }
//     };

//     checkElectronAPI();
//   }, []);

//   return (
//     <div style={{ padding: "2rem", fontFamily: "sans-serif" }}>
//       <h1>Hello MoodMap Desktop!</h1>
//     </div>
//   );
// }
export default App;
