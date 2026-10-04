import "./styles/globals.css";
import { initTheme } from "./lib/theme";
import { App } from "./app/app";

initTheme();
const app = new App(document.getElementById("app")!);
void app.boot();
(window as unknown as { onnxviz: App }).onnxviz = app;
