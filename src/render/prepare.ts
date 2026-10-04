import type { LayoutOptions, ModelView } from "../onnx/types";
import { layoutScene } from "../workers/layout-client";
import { buildScene, type Scene, type SceneOptions } from "./scene";
import { readTheme } from "./theme";

/** make sure the canvas fonts used for measuring are actually loaded before building a scene */
async function fontsReady(sans: string): Promise<void> {
  try {
    await Promise.all([document.fonts.load(`600 12px ${sans}`), document.fonts.load(`10.5px ${sans}`)]);
    await document.fonts.ready;
  } catch { /* fonts API unavailable */ }
}

/**
 * One call for the app shell: build the scene for a graph with the current theme's fonts,
 * run layout in the worker and return it ready for `viewer.setScene()`.
 * Rejects with LayoutCancelled if a newer prepareScene()/layoutScene() call supersedes it.
 */
export async function prepareScene(model: ModelView, graphId: number, opts: Partial<SceneOptions> = {}, layout?: Partial<LayoutOptions>): Promise<Scene> {
  const th = readTheme();
  await fontsReady(th.fontSans);
  const scene = buildScene(model, graphId, opts, { sans: th.fontSans, mono: th.fontMono });
  if (scene.nodes.length) await layoutScene(scene, layout);
  else scene.laidOut = true;
  return scene;
}
