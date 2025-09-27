// Electron main process
import { app, BrowserWindow } from "electron";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// checking 
// console.log("Electron main.js has started");


function createWindow() {
    const mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            preload: path.join(__dirname, "preload.js"),
            contextIsolation: true,
            nodeIntegration: false,  
        },
    });

    //running in dev / preview / production accordingly 
    if (app.isPackaged) {
        // In production, load the local HTML file
        mainWindow.loadFile(path.join(__dirname, "../dist/index.html"));
    }else{
        // In development, load the Vite dev server URL
        mainWindow.loadURL("http://localhost:5173"); 
        mainWindow.webContents.openDevTools();
    }
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
    createWindow();

    app.on("activate", () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
})


app.on("window-all-closed", () => {
    if (process.platform !== "darwin") app.quit();
});