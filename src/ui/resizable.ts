import { h } from "../lib/dom";
import { cn } from "../lib/cn";

/** Attach a pointer-drag + keyboard resize behaviour to a handle element. */
export function dragResize(handle: HTMLElement, o: { axis: "x" | "y"; get(): number; set(v: number): void; min: number; max(): number; invert?: boolean; onEnd?: (v: number) => void }): void {
  handle.setAttribute("role", "separator"); handle.tabIndex = 0;
  handle.setAttribute("aria-orientation", o.axis === "x" ? "vertical" : "horizontal");
  const clamp = (v: number) => Math.max(o.min, Math.min(o.max(), v));
  handle.addEventListener("pointerdown", (e) => {
    e.preventDefault(); handle.setPointerCapture(e.pointerId);
    const start = o.axis === "x" ? e.clientX : e.clientY, v0 = o.get(), sgn = o.invert ? -1 : 1;
    handle.dataset.dragging = "";
    document.body.style.userSelect = "none"; document.body.style.cursor = o.axis === "x" ? "col-resize" : "row-resize";
    const move = (ev: PointerEvent) => o.set(clamp(v0 + sgn * ((o.axis === "x" ? ev.clientX : ev.clientY) - start)));
    const up = () => { handle.removeEventListener("pointermove", move); handle.removeEventListener("pointerup", up); delete handle.dataset.dragging; document.body.style.userSelect = ""; document.body.style.cursor = ""; o.onEnd?.(o.get()); };
    handle.addEventListener("pointermove", move); handle.addEventListener("pointerup", up);
  });
  handle.addEventListener("keydown", (e) => {
    const k = o.axis === "x" ? ["ArrowLeft", "ArrowRight"] : ["ArrowUp", "ArrowDown"];
    const d = e.key === k[0] ? -1 : e.key === k[1] ? 1 : 0;
    if (!d) return;
    e.preventDefault(); o.set(clamp(o.get() + d * (o.invert ? -1 : 1) * (e.shiftKey ? 48 : 12))); o.onEnd?.(o.get());
  });
}

export interface ResizableCtl { el: HTMLElement; setSize(px: number): void; getSize(): number }

/** Two panes with a draggable divider. `a` has a controlled size (px); `b` flexes. */
export function resizablePanels(p: { direction?: "horizontal" | "vertical"; a: Node; b: Node; size?: number; min?: number; maxFrac?: number; storageKey?: string; class?: string }): ResizableCtl {
  const horiz = (p.direction ?? "horizontal") === "horizontal";
  let size = p.size ?? 280;
  if (p.storageKey) try { const v = Number(localStorage.getItem(p.storageKey)); if (v > 0) size = v; } catch { /* ignore */ }
  const pa = h("div", { class: "min-h-0 min-w-0 overflow-hidden shrink-0" }, p.a);
  const pb = h("div", { class: "min-h-0 min-w-0 flex-1 overflow-hidden" }, p.b);
  const handle = h("div", { "data-slot": "resizable-handle", class: cn("bg-border focus-visible:ring-ring relative flex shrink-0 items-center justify-center outline-hidden focus-visible:ring-1 data-[dragging]:bg-primary hover:bg-primary/60 transition-colors after:absolute after:inset-y-0 after:left-1/2 after:w-1 after:-translate-x-1/2", horiz ? "w-px cursor-col-resize after:w-2" : "h-px cursor-row-resize after:inset-x-0 after:h-2 after:w-full after:left-0 after:translate-x-0") });
  const el = h("div", { "data-slot": "resizable", class: cn("flex h-full w-full", horiz ? "flex-row" : "flex-col", p.class) }, pa, handle, pb);
  const apply = (v: number) => { size = v; horiz ? (pa.style.width = `${v}px`) : (pa.style.height = `${v}px`); };
  dragResize(handle, {
    axis: horiz ? "x" : "y", get: () => size, set: apply, min: p.min ?? 120,
    max: () => Math.max(p.min ?? 120, (horiz ? el.clientWidth : el.clientHeight) * (p.maxFrac ?? 0.8)),
    onEnd: (v) => { if (p.storageKey) try { localStorage.setItem(p.storageKey, String(Math.round(v))); } catch { /* ignore */ } },
  });
  apply(size);
  return { el, setSize: apply, getSize: () => size };
}
