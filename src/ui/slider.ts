import { h } from "../lib/dom";
import { cn } from "../lib/cn";

export interface SliderCtl { el: HTMLElement; get(): number; set(v: number, silent?: boolean): void }
export function slider(p: { min?: number; max?: number; step?: number; value?: number; disabled?: boolean; ariaLabel?: string; class?: string; onInput?: (v: number) => void; onCommit?: (v: number) => void } = {}): SliderCtl {
  const min = p.min ?? 0, max = p.max ?? 100, step = p.step ?? 1;
  let val = p.value ?? min;
  const range = h("span", { "data-slot": "slider-range", class: "bg-primary absolute h-full" });
  const track = h("span", { "data-slot": "slider-track", class: "bg-muted relative h-1.5 w-full grow overflow-hidden rounded-full" }, range);
  const thumb = h("span", {
    "data-slot": "slider-thumb", role: "slider", tabindex: p.disabled ? -1 : 0, "aria-label": p.ariaLabel, "aria-valuemin": min, "aria-valuemax": max,
    class: "border-primary bg-background ring-ring/50 absolute top-1/2 block size-4 shrink-0 -translate-x-1/2 -translate-y-1/2 rounded-full border shadow-sm transition-[color,box-shadow] hover:ring-4 focus-visible:ring-4 focus-visible:outline-hidden",
  });
  const el = h("span", { "data-slot": "slider", "data-disabled": p.disabled ? "" : undefined, class: cn("relative flex w-full touch-none items-center py-2 select-none data-[disabled]:opacity-50", p.class) }, track, thumb);
  const clamp = (v: number) => { const s = Math.round((v - min) / step) * step + min; return Math.min(max, Math.max(min, +s.toFixed(10))); };
  const set = (v: number, silent = false) => {
    val = clamp(v);
    const f = max > min ? ((val - min) / (max - min)) * 100 : 0;
    range.style.width = `${f}%`; thumb.style.left = `${f}%`;
    thumb.setAttribute("aria-valuenow", String(val));
    if (!silent) p.onInput?.(val);
  };
  const fromPointer = (e: PointerEvent) => { const r = track.getBoundingClientRect(); set(min + ((e.clientX - r.left) / r.width) * (max - min)); };
  el.addEventListener("pointerdown", (e) => {
    if (p.disabled) return;
    el.setPointerCapture(e.pointerId); thumb.focus(); fromPointer(e);
    const move = (ev: PointerEvent) => fromPointer(ev);
    const up = () => { el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); p.onCommit?.(val); };
    el.addEventListener("pointermove", move); el.addEventListener("pointerup", up);
  });
  thumb.addEventListener("keydown", (e) => {
    const big = Math.max(step, (max - min) / 10);
    const d: Record<string, number> = { ArrowRight: step, ArrowUp: step, ArrowLeft: -step, ArrowDown: -step, PageUp: big, PageDown: -big };
    if (e.key in d) { e.preventDefault(); set(val + d[e.key]!); p.onCommit?.(val); }
    else if (e.key === "Home") { e.preventDefault(); set(min); p.onCommit?.(val); }
    else if (e.key === "End") { e.preventDefault(); set(max); p.onCommit?.(val); }
  });
  set(val, true);
  return { el, get: () => val, set };
}
