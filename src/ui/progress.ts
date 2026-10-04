import { h } from "../lib/dom";
import { cn } from "../lib/cn";

export interface ProgressCtl { el: HTMLElement; set(v: number | null): void }
/** value 0..100, null = indeterminate */
export function progress(value: number | null = 0, cls?: string): ProgressCtl {
  const bar = h("div", { "data-slot": "progress-indicator", class: "bg-primary h-full w-full flex-1 transition-all" });
  const el = h("div", { "data-slot": "progress", role: "progressbar", "aria-valuemin": 0, "aria-valuemax": 100, class: cn("bg-primary/20 relative h-2 w-full overflow-hidden rounded-full", cls) }, bar);
  const set = (v: number | null) => {
    if (v == null) { el.removeAttribute("aria-valuenow"); bar.style.transform = ""; bar.classList.add("ui-indeterminate", "!w-1/3"); }
    else { bar.classList.remove("ui-indeterminate", "!w-1/3"); el.setAttribute("aria-valuenow", String(Math.round(v))); bar.style.transform = `translateX(-${100 - Math.max(0, Math.min(100, v))}%)`; }
  };
  set(value);
  return { el, set };
}
