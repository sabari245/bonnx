// Electron main process: loads the single-file build from disk, handles file association / CLI paths.
const { app, BrowserWindow, dialog, ipcMain, Menu, shell, nativeTheme } = require("electron");
const fs = require("node:fs");
const path = require("node:path");

const INDEX = path.join(__dirname, "..", "dist-single", "index.html");
const EXTS = /\.(onnx|ort|pb|prototxt|pbtxt|json|txt)$/i;
const SIBLING = /\.(bin|data|weights|pb)$/i;
const MAX_SIBLING = 4 * 1024 ** 3;

let win = null;
let pending = null; // path to open once the renderer is ready
let ready = false;

const pathFromArgs = (argv) => argv.slice(app.isPackaged ? 1 : 2).find((a) => !a.startsWith("-") && EXTS.test(a) && fs.existsSync(a));

async function readModel(file) {
  const dir = path.dirname(file), base = path.basename(file), stem = base.replace(/\.[^.]+$/, "");
  const model = await fs.promises.readFile(file);
  const external = {};
  try {
    for (const e of await fs.promises.readdir(dir, { withFileTypes: true })) {
      if (!e.isFile() || e.name === base || /\.onnx$/i.test(e.name)) continue;
      if (!(e.name.startsWith(stem) || SIBLING.test(e.name))) continue;
      const p = path.join(dir, e.name);
      if ((await fs.promises.stat(p)).size > MAX_SIBLING) continue;
      external[e.name] = await fs.promises.readFile(p);
    }
  } catch { /* external data is optional */ }
  return { name: base, data: model, external };
}

function send(file) {
  if (!win || !ready) { pending = file; return; }
  win.webContents.send("open-path", file);
  if (win.isMinimized()) win.restore();
  win.focus();
}

function createWindow() {
  win = new BrowserWindow({
    width: 1440, height: 900, minWidth: 720, minHeight: 480,
    title: "Bonnx", backgroundColor: nativeTheme.shouldUseDarkColors ? "#0a0a0a" : "#ffffff",
    icon: path.join(__dirname, "..", "build", "icon.png"),
    webPreferences: { preload: path.join(__dirname, "preload.cjs"), contextIsolation: true, sandbox: true, nodeIntegration: false },
  });
  win.removeMenu?.();
  win.loadFile(INDEX);
  win.webContents.setWindowOpenHandler(({ url }) => { if (/^https?:/.test(url)) shell.openExternal(url); return { action: "deny" }; });
  win.webContents.on("will-navigate", (e, url) => { if (!url.startsWith("file:")) { e.preventDefault(); shell.openExternal(url); } });
  win.on("closed", () => { win = null; ready = false; });
}

ipcMain.handle("ready", () => { ready = true; const p = pending; pending = null; return p; });
ipcMain.handle("read-model", (_e, file) => readModel(file));
ipcMain.handle("pick-model", async () => {
  const r = await dialog.showOpenDialog(win, { title: "Open ONNX model", properties: ["openFile"], filters: [{ name: "ONNX models", extensions: ["onnx", "ort", "pb", "prototxt", "pbtxt", "json"] }, { name: "All files", extensions: ["*"] }] });
  return r.canceled ? null : r.filePaths[0];
});

if (!app.requestSingleInstanceLock()) app.quit();
else {
  app.on("second-instance", (_e, argv) => { const f = pathFromArgs(argv); if (f) send(path.resolve(f)); else if (win) { if (win.isMinimized()) win.restore(); win.focus(); } });
  app.on("open-file", (e, f) => { e.preventDefault(); send(f); });
  app.whenReady().then(() => {
    Menu.setApplicationMenu(null);
    const f = pathFromArgs(process.argv);
    if (f) pending = path.resolve(f);
    createWindow();
  });
  app.on("window-all-closed", () => app.quit());
}
