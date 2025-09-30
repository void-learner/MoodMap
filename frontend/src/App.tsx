import React, { useEffect, useState } from "react";
import ChatBubble from "./components/ChatBubble";
import InputField from "./components/InputField";
import EmotionFeedback from "./components/EmotionFeedback";

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
      <header className="bg-white shadow">
        <h1 className="text-3xl font-bold text-center py-6">MoodMap Desktop</h1>
      </header>
      {/* <Chatbot /> */}
    </div>
  );
};




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
//     console.log("Message sent:", input); // 🔥 check if message is captured
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
//       <ChatBubble message={{ text: "Hello! I'm the bot 🤖", sender: "bot" }} />
//       <ChatBubble message={{ text: "Hi there! I'm the user 🙋‍♀️", sender: "user" }} />
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
