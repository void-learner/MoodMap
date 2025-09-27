const { contextBridge } = require('electron');

// checking
console.log("preload.js is running");

// Expose a safe API to the renderer process
contextBridge.exposeInMainWorld("electronAPI", {
  ping: () => console.log("Ping from preload"),
});


// contextBridge.exposeInMainWorld('version', {
//     node: () => process.versions.node,
//     chrome: () => process.versions.chrome,
//     electron: () => process.versions.electron
// });