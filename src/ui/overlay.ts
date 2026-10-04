import { h } from "../lib/dom";

/** Shared infrastructure for floating layers: portal root, positioning, dismissal and focus helpers. */

export type Side = "top" | "bottom" | "left" | "right";
export type Align = "start" | "center" | "end";

let root: HTMLElement | null = null;
export function portalRoot(): HTMLElement {
  if (!root || !root.isConnected) {
    root = h("div", { id: "ui-portal", class: "contents" });
    document.body.appendChild(root);
  }
  return root;
}

export const reducedMotion = (): boolean => matchMedia("(prefers-reduced-motion: reduce)").matches;

export interface PositionOpts {
  side?: Side;
  align?: Align;
  sideOffset?: number;
  alignOffset?: number;
  /** match the anchor width (select / combobox) */
  matchWidth?: boolean;
  collisionPadding?: number;
}

/** Position `el` (position:fixed) next to `anchor` (element or point), flipping when it would overflow. Returns the resolved side. */
export function position(el: HTMLElement, anchor: Element | { x: number; y: number }, o: PositionOpts = {}): Side {
  const pad = o.collisionPadding ?? 8, off = o.sideOffset ?? 6;
  const a = anchor instanceof Element ? anchor.getBoundingClientRect() : new DOMRect(anchor.x, anchor.y, 0, 0);
  if (o.matchWidth) el.style.minWidth = `${a.width}px`;
  el.style.left = "0px";
  el.style.top = "0px";
  const r = el.getBoundingClientRect();
  const vw = innerWidth, vh = innerHeight;
  let side: Side = o.side ?? "bottom";
  const room = { top: a.top - pad, bottom: vh - a.bottom - pad, left: a.left - pad, right: vw - a.right - pad };
  const need = side === "top" || side === "bottom" ? r.height + off : r.width + off;
  const opp: Record<Side, Side> = { top: "bottom", bottom: "top", left: "right", right: "left" };
  if (room[side] < need && room[opp[side]] > room[side]) side = opp[side];
  let x: number, y: number;
  const al = o.align ?? "center", ao = o.alignOffset ?? 0;
  if (side === "top" || side === "bottom") {
    y = side === "bottom" ? a.bottom + off : a.top - r.height - off;
    x = al === "start" ? a.left + ao : al === "end" ? a.right - r.width - ao : a.left + a.width / 2 - r.width / 2;
  } else {
    x = side === "right" ? a.right + off : a.left - r.width - off;
    y = al === "start" ? a.top + ao : al === "end" ? a.bottom - r.height - ao : a.top + a.height / 2 - r.height / 2;
  }
  x = Math.max(pad, Math.min(x, vw - r.width - pad));
  y = Math.max(pad, Math.min(y, vh - r.height - pad));
  el.style.left = `${Math.round(x)}px`;
  el.style.top = `${Math.round(y)}px`;
  el.dataset.side = side;
  return side;
}

/** Layer stack: Escape / outside pointerdown dismiss only the topmost layer. */
interface Layer { el: HTMLElement; close: () => void; ignore: (t: Node) => boolean }
const stack: Layer[] = [];
let wired = false;
function wire(): void {
  if (wired) return;
  wired = true;
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && stack.length) {
      e.stopPropagation();
      e.preventDefault();
      stack[stack.length - 1]!.close();
    }
  }, true);
  document.addEventListener("pointerdown", (e) => {
    const top = stack[stack.length - 1];
    if (!top) return;
    const t = e.target as Node;
    if (top.el.contains(t) || top.ignore(t)) return;
    // a click inside a parent-owned child layer is handled by that layer; here we just close the top
    top.close();
  }, true);
}
/** Register a dismissable layer; returns an unregister function. `ignore` marks nodes (e.g. the trigger) that must not count as outside. */
export function pushLayer(el: HTMLElement, close: () => void, ignore: (t: Node) => boolean = () => false): () => void {
  wire();
  const l = { el, close, ignore };
  stack.push(l);
  return () => {
    const i = stack.indexOf(l);
    if (i >= 0) stack.splice(i, 1);
  };
}

const FOCUSABLE = 'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])';
export const focusables = (el: HTMLElement): HTMLElement[] => [...el.querySelectorAll<HTMLElement>(FOCUSABLE)].filter((x) => x.offsetParent !== null || x === document.activeElement);

/** Trap Tab inside `el`. Returns a release function that restores focus to the previously focused element. */
export function trapFocus(el: HTMLElement): () => void {
  const prev = document.activeElement as HTMLElement | null;
  const onKey = (e: KeyboardEvent) => {
    if (e.key !== "Tab") return;
    const f = focusables(el);
    if (!f.length) { e.preventDefault(); el.focus(); return; }
    const first = f[0]!, last = f[f.length - 1]!;
    if (e.shiftKey && (document.activeElement === first || document.activeElement === el)) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  };
  el.addEventListener("keydown", onKey);
  return () => {
    el.removeEventListener("keydown", onKey);
    prev?.focus?.({ preventScroll: true });
  };
}

/** Set data-state=closed, wait for the exit animation, then remove. */
export function exit(el: HTMLElement, done?: () => void): void {
  el.dataset.state = "closed";
  const fin = () => { el.remove(); done?.(); };
  if (reducedMotion() || !el.getAnimations().length) return fin();
  Promise.allSettled(el.getAnimations().map((a) => a.finished)).then(fin);
}

let uid = 0;
export const nextId = (p = "ui"): string => `${p}-${++uid}`;
