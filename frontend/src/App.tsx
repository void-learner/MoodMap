import React, { useEffect } from "react";

// Extend the Window interface to include electronAPI
declare global {
  interface Window {
    electronAPI?: {
      ping: () => void;
    };
  }
}

function App() {
  useEffect(() => {
    // Function to check if electronAPI is available
    const checkElectronAPI = () => {
      if (window.electronAPI) {
        console.log("electronAPI is available in React");
        window.electronAPI.ping(); // logs "Ping from preload" in terminal
      } else {
        console.log("electronAPI NOT found, retrying...");
      }
    };

    checkElectronAPI();
  }, []);

  return (
    <div style={{ padding: "2rem", fontFamily: "sans-serif" }}>
      <h1>Hello MoodMap Desktop!</h1>
    </div>
  );
}

export default App;
