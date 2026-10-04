/** Canvas theme: reads the CSS custom properties defined by the design system, with built-in fallbacks. */

export type RGBA = [number, number, number, number];

export interface Theme {
  dark: boolean;
  bg: string; edge: string; edgeHi: string; nodeBg: string; nodeRow: string; nodeBorder: string;
  nodeText: string; nodeSub: string; grey: string; greyText: string; sel: string; grid: string; match: string;
  cats: Record<CatKey, string>;
  ramp: string[]; // 5 stops
  fontSans: string; fontMono: string;
}

export type CatKey = "layer" | "activation" | "norm" | "pool" | "math" | "reduce" | "shape" | "quant" | "control" | "other" | "input" | "const";

const LIGHT: Record<string, string> = {
  "--graph-bg": "#ffffff", "--graph-edge": "#71717a", "--graph-edge-hi": "#2563eb", "--graph-node-bg": "#fafafa",
  "--graph-node-row": "#f4f4f5", "--graph-node-border": "#a1a1aa", "--graph-node-text": "#27272a", "--graph-node-subtext": "#71717a",
  "--graph-grey": "#d4d4d8", "--graph-grey-text": "#52525b", "--graph-sel": "#2563eb", "--graph-grid": "#e4e4e7",
  "--cat-layer": "#3a6ea5", "--cat-activation": "#b5433c", "--cat-norm": "#3d7c4d", "--cat-pool": "#2f7f86", "--cat-math": "#6c6f78",
  "--cat-reduce": "#7b5aa6", "--cat-shape": "#8a6d3b", "--cat-quant": "#b8741a", "--cat-control": "#a23b72", "--cat-other": "#8e8e93",
  "--cat-input": "#27272a", "--cat-const": "#a1a1aa",
  "--ramp-0": "#deebf7", "--ramp-1": "#9ecae1", "--ramp-2": "#4292c6", "--ramp-3": "#08519c", "--ramp-4": "#08306b",
};
const DARK: Record<string, string> = {
  ...LIGHT,
  "--graph-bg": "#18181b", "--graph-edge": "#a1a1aa", "--graph-edge-hi": "#60a5fa", "--graph-node-bg": "#27272a",
  "--graph-node-row": "#2f2f33", "--graph-node-border": "#52525b", "--graph-node-text": "#e4e4e7", "--graph-node-subtext": "#a1a1aa",
  "--graph-grey": "#3f3f46", "--graph-grey-text": "#d4d4d8", "--graph-sel": "#60a5fa", "--graph-grid": "#27272a",
  "--cat-input": "#d4d4d8", "--cat-const": "#52525b",
  "--cat-layer": "#4a83c2", "--cat-activation": "#c75a52", "--cat-norm": "#4d9461", "--cat-pool": "#3b98a0", "--cat-math": "#80848f",
  "--cat-reduce": "#9072bf", "--cat-shape": "#a3834a", "--cat-quant": "#cf8a2a", "--cat-control": "#c24f8a", "--cat-other": "#9c9ca3",
  "--ramp-0": "#1e3a5f", "--ramp-1": "#2563a8", "--ramp-2": "#4a9ae0", "--ramp-3": "#9ccaf5", "--ramp-4": "#e0f0ff",
};

let probe: CanvasRenderingContext2D | null = null;
const ctx1 = () => (probe ??= document.createElement("canvas").getContext("2d", { willReadFrequently: true })!);

/** Convert any CSS colour (hex, rgb, oklch, hsl, color-mix …) to [r,g,b,a] via a 1px canvas round trip. */
export function parseColor(css: string): RGBA {
  const c = ctx1();
  c.canvas.width = c.canvas.height = 1;
  c.clearRect(0, 0, 1, 1);
  c.fillStyle = "#000";
  c.fillStyle = css;
  c.fillRect(0, 0, 1, 1);
  const d = c.getImageData(0, 0, 1, 1).data;
  return [d[0], d[1], d[2], d[3] / 255];
}
export const toCss = (c: RGBA): string => (c[3] >= 1 ? `rgb(${c[0]},${c[1]},${c[2]})` : `rgba(${c[0]},${c[1]},${c[2]},${+c[3].toFixed(3)})`);
export const luminance = (c: RGBA): number => (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255;

export function isDark(): boolean {
  const el = document.documentElement;
  if (el.classList.contains("dark")) return true;
  if (el.classList.contains("light")) return false;
  if (el.dataset.theme) return el.dataset.theme === "dark";
  return matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
}

export function readTheme(): Theme {
  const dark = isDark();
  const fb = dark ? DARK : LIGHT;
  const cs = getComputedStyle(document.documentElement);
  const g = (k: string): string => {
    const v = cs.getPropertyValue(k).trim();
    return toCss(parseColor(v || fb[k]));
  };
  const font = (k: string, fallback: string) => {
    const v = cs.getPropertyValue(k).trim();
    return v ? `${v}, ${fallback}` : fallback;
  };
  const cats = {} as Record<CatKey, string>;
  for (const k of ["layer", "activation", "norm", "pool", "math", "reduce", "shape", "quant", "control", "other", "input", "const"] as CatKey[]) cats[k] = g(`--cat-${k}`);
  return {
    dark, bg: g("--graph-bg"), edge: g("--graph-edge"), edgeHi: g("--graph-edge-hi"), nodeBg: g("--graph-node-bg"),
    nodeRow: g("--graph-node-row"), nodeBorder: g("--graph-node-border"), nodeText: g("--graph-node-text"),
    nodeSub: g("--graph-node-subtext"), grey: g("--graph-grey"), greyText: g("--graph-grey-text"), sel: g("--graph-sel"),
    grid: g("--graph-grid"), match: dark ? "#fbbf24" : "#d97706", cats,
    ramp: [0, 1, 2, 3, 4].map((i) => g(`--ramp-${i}`)),
    fontSans: font("--font-sans", 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'),
    fontMono: font("--font-mono", 'ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace'),
  };
}

/** text colour (white or near-black) that is legible on the given fill */
export function textOn(fill: string, theme?: Theme): string {
  void theme;
  return luminance(parseColor(fill)) > 0.55 ? "#18181b" : "#ffffff";
}

/** notify canvases that the theme changed (call after toggling the `dark` class) */
export const notifyThemeChange = (): boolean => window.dispatchEvent(new Event("themechange"));
