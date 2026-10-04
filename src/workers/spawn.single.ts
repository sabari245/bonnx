import ModelWorker from "./model.worker.ts?worker&inline";
import LayoutWorker from "./layout.worker.ts?worker&inline";

export const spawnModelWorker = (): Worker => new ModelWorker();
export const spawnLayoutWorker = (): Worker => new LayoutWorker();
