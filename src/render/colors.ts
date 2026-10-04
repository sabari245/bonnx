import { DTYPE_CATEGORIES, fmtCount, fmtNum, type DTypeCategory } from "../lib/format";
import { OP_CATEGORIES, type RNode, type Scene } from "./scene";
import { parseColor, textOn, toCss, luminance, type CatKey, type Theme } from "./theme";

export type ColorMode = "op" | "params" | "wmax" | "dtype" | "depth";
export const COLOR_MODES: { key: ColorMode; label: string; hint: string }[] = [
  { key: "op", label: "Operator type", hint: "Colour by operator category" },
  { key: "params", label: "Parameters", hint: "Colour by number of float weights (log scale)" },
  { key: "wmax", label: "Largest |w|", hint: "Colour by the largest absolute weight (log scale)" },
  { key: "dtype", label: "Data type", hint: "Colour by output data type" },
  { key: "depth", label: "Depth", hint: "Colour by topological depth" },
];

export interface Paint { fill: string; text: string }
export interface LegendEntry { label: string; color: string; count: number }
export interface Legend {
  title: string;
  kind: "categorical" | "ramp";
  entries: LegendEntry[];
  ramp?: { stops: string[]; lo: string; mid: string; hi: string; none: number };
}

const GREY = -1, IO = -3, CONST = -4, GHOST = -5;
const NB = 48;
const DTYPE_CAT_COLOR: Record<DTypeCategory, CatKey> = { float: "layer", lowfloat: "quant", int: "reduce", int8: "shape", bool: "math", string: "norm", other: "other" };

function rampPalette(theme: Theme): Paint[] {
  const stops = theme.ramp.map(parseColor);
  const out: Paint[] = [];
  for (let i = 0; i < NB; i++) {
    const t = (i / (NB - 1)) * (stops.length - 1), k = Math.min(stops.length - 2, Math.floor(t)), f = t - k;
    const c = [0, 1, 2].map((j) => Math.round(stops[k][j] + (stops[k + 1][j] - stops[k][j]) * f));
    const fill = toCss([c[0], c[1], c[2], 1]);
    out.push({ fill, text: luminance([c[0], c[1], c[2], 1]) > 0.55 ? "#18181b" : "#ffffff" });
  }
  return out;
}

/** Per-(scene, theme, mode) colour assignment: a small int code per node + palette lookup. */
export class ColorScheme {
  readonly codes: Int16Array;
  private palette: Paint[];
  private special: Record<number, Paint>;
  private lo = 0; private hi = 0;
  private cats: string[] = [];

  constructor(readonly scene: Scene, readonly theme: Theme, readonly mode: ColorMode) {
    const N = scene.nodes;
    this.codes = new Int16Array(N.length);
    const mk = (fill: string): Paint => ({ fill, text: textOn(fill) });
    this.special = {
      [GREY]: { fill: theme.grey, text: theme.greyText },
      [IO]: mk(theme.cats.input),
      [CONST]: { fill: theme.cats.const, text: textOn(theme.cats.const) },
      [GHOST]: { fill: theme.grey, text: theme.greyText },
    };
    if (mode === "op") {
      this.palette = OP_CATEGORIES.map((c) => mk(theme.cats[c.key]));
      this.cats = OP_CATEGORIES.map((c) => c.label);
    } else if (mode === "dtype") {
      this.palette = DTYPE_CATEGORIES.map((c) => mk(theme.cats[DTYPE_CAT_COLOR[c.key]]));
      this.cats = DTYPE_CATEGORIES.map((c) => c.label);
    } else this.palette = rampPalette(theme);

    const ops = N.filter((n) => n.kind === "op" || n.kind === "fn");
    const logOf = (get: (n: RNode) => number): void => {
      const v = ops.map(get).filter((x) => x > 0);
      this.lo = v.length ? Math.log10(v.reduce((a, b) => Math.min(a, b))) : 0;
      this.hi = v.length ? Math.log10(v.reduce((a, b) => Math.max(a, b))) : 0;
    };
    const bucket = (t: number): number => Math.round(Math.min(1, Math.max(0, t)) * (NB - 1));
    if (mode === "params") logOf((n) => n.params);
    else if (mode === "wmax") logOf((n) => n.wmax ?? 0);
    const lt = (x: number): number => (this.hi > this.lo ? (Math.log10(x) - this.lo) / (this.hi - this.lo) : 1);

    for (const n of N) {
      let c: number;
      switch (n.kind) {
        case "input": case "output": c = IO; break;
        case "const": c = CONST; break;
        case "ghost": c = GHOST; break;
        default:
          if (mode === "op") c = n.cat;
          else if (mode === "dtype") c = n.dtypeCat ? DTYPE_CATEGORIES.findIndex((d) => d.key === n.dtypeCat) : GREY;
          else if (mode === "params") c = n.params > 0 ? bucket(lt(n.params)) : GREY;
          else if (mode === "wmax") c = n.wmax != null && n.wmax > 0 ? bucket(lt(n.wmax)) : GREY;
          else c = bucket(scene.maxDepth ? n.depth / scene.maxDepth : 0);
      }
      this.codes[n.id] = c;
    }
  }

  paintOfCode(c: number): Paint { return c >= 0 ? this.palette[c] : this.special[c]; }
  paint(id: number): Paint { return this.paintOfCode(this.codes[id]); }

  metricText(n: RNode): string {
    if (n.kind !== "op" && n.kind !== "fn") return "";
    switch (this.mode) {
      case "params": return n.params ? `${fmtCount(n.params)} params` : "no weights";
      case "wmax": return n.wmax != null ? `|w| max ${fmtNum(n.wmax, 3)}` : "no weights";
      case "depth": return `depth ${n.depth}`;
      default: return n.outType ? `→ ${n.outType}` : "";
    }
  }

  legend(): Legend {
    const N = this.scene.nodes.filter((n) => n.kind === "op" || n.kind === "fn");
    if (this.mode === "op" || this.mode === "dtype") {
      const cnt = new Map<number, number>();
      for (const n of N) cnt.set(this.codes[n.id], (cnt.get(this.codes[n.id]) ?? 0) + 1);
      const entries: LegendEntry[] = [];
      this.palette.forEach((p, i) => cnt.get(i) && entries.push({ label: this.cats[i], color: p.fill, count: cnt.get(i)! }));
      if (cnt.get(GREY)) entries.push({ label: "unknown", color: this.special[GREY].fill, count: cnt.get(GREY)! });
      return { title: this.mode === "op" ? "Operator type" : "Output data type", kind: "categorical", entries };
    }
    const none = N.filter((n) => this.codes[n.id] === GREY).length;
    const stops = this.theme.ramp;
    if (this.mode === "depth") {
      const m = this.scene.maxDepth;
      return { title: "Topological depth", kind: "ramp", entries: [], ramp: { stops, lo: "0", mid: String(Math.round(m / 2)), hi: String(m), none } };
    }
    const f = this.mode === "params" ? (v: number) => fmtCount(Math.round(v)) : (v: number) => fmtNum(v, 3);
    const at = (t: number) => f(10 ** (this.lo + t * (this.hi - this.lo)));
    return { title: this.mode === "params" ? "Parameters per layer (log)" : "Largest |weight| per layer (log)", kind: "ramp", entries: [], ramp: { stops, lo: at(0), mid: at(0.5), hi: at(1), none } };
  }
}
