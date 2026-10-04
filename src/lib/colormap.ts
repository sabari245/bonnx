/** 256-entry colormap LUTs (interleaved RGB) + value normalization helpers for the tensor image view. */

export type ColormapName = "gray" | "viridis" | "magma" | "coolwarm";

export const COLORMAPS: { value: ColormapName; label: string; diverging?: boolean }[] = [
  { value: "gray", label: "Grayscale" },
  { value: "viridis", label: "Viridis" },
  { value: "magma", label: "Magma" },
  { value: "coolwarm", label: "Cool–warm (diverging)", diverging: true },
];

const STOPS: Record<ColormapName, string[]> = {
  gray: ["#000000", "#ffffff"],
  viridis: ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"],
  magma: ["#000004", "#140e36", "#3b0f70", "#641a80", "#8c2981", "#b73779", "#de4968", "#f7705c", "#fe9f6d", "#fcfdbf"],
  coolwarm: ["#3b4cc0", "#8db0fe", "#dddddd", "#f49a7b", "#b40426"],
};

const luts = new Map<ColormapName, Uint8ClampedArray>();

/** 256 × RGB entries. */
export function colormapLUT(name: ColormapName): Uint8ClampedArray {
  let lut = luts.get(name);
  if (lut) return lut;
  const rgb = STOPS[name].map((s) => [1, 3, 5].map((i) => parseInt(s.slice(i, i + 2), 16)));
  lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (rgb.length - 1);
    const k = Math.min(rgb.length - 2, Math.floor(t));
    const f = t - k;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(rgb[k]![c]! + (rgb[k + 1]![c]! - rgb[k]![c]!) * f);
  }
  luts.set(name, lut);
  return lut;
}

/** CSS color for t in [0,1], handy for legends. */
export function colormapCSS(name: ColormapName, t: number): string {
  const l = colormapLUT(name);
  const i = Math.max(0, Math.min(255, Math.round(t * 255))) * 3;
  return `rgb(${l[i]},${l[i + 1]},${l[i + 2]})`;
}

export function colormapGradient(name: ColormapName): string {
  const n = 8;
  return `linear-gradient(90deg, ${Array.from({ length: n + 1 }, (_, i) => colormapCSS(name, i / n)).join(", ")})`;
}

export type NormMode = "minmax" | "tile" | "symmetric" | "percentile" | "custom";

export const NORM_MODES: { value: NormMode; label: string; hint: string }[] = [
  { value: "minmax", label: "Min–max", hint: "Whole-tensor minimum → maximum" },
  { value: "tile", label: "Per tile", hint: "Each tile normalized to its own minimum and maximum" },
  { value: "symmetric", label: "Symmetric", hint: "−max|x| … +max|x| so zero is the midpoint" },
  { value: "percentile", label: "P1 – P99", hint: "Clip outliers: 1st to 99th percentile" },
  { value: "custom", label: "Custom", hint: "Enter your own range" },
];

export const NAN_COLOR: [number, number, number] = [255, 0, 220];
export const INF_COLOR: [number, number, number] = [255, 160, 0];
