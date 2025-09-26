const { contextBridge } = require("electron");

// checking
console.log("preload.js is running");

// Expose a safe API to the renderer process
contextBridge.exposeInMainWorld("electronAPI", {
    ping: () => console.log("Ping from preload")
});