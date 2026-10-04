const { contextBridge, ipcRenderer } = require("electron");
contextBridge.exposeInMainWorld("bonnxDesktop", {
  ready: () => ipcRenderer.invoke("ready"),
  readModel: (p) => ipcRenderer.invoke("read-model", p),
  pickModel: () => ipcRenderer.invoke("pick-model"),
  onOpenPath: (cb) => ipcRenderer.on("open-path", (_e, p) => cb(p)),
});
