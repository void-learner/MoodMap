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
        // Retry after 50ms in case preload is not ready yet
        setTimeout(checkElectronAPI, 50);
      }
    };

    checkElectronAPI();
  }, []);

  return (
    <div style={{ padding: "2rem", fontFamily: "sans-serif" }}>
      <h1>Hello MoodMap Desktop!</h1>
      <p>This is your Electron + React + Vite app running.</p>
      <p>Check the console and terminal for logs from preload and React.</p>
    </div>
  );
}

export default App;
