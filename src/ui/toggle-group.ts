import { h, icon } from "../lib/dom";
import { cn } from "../lib/cn";
import { toggleVariants } from "./toggle";
import type { IconNode } from "lucide";

export interface ToggleItem { value: string; label?: string | Node; icon?: IconNode; title?: string; ariaLabel?: string; disabled?: boolean }
export interface ToggleGroupCtl { el: HTMLElement; get(): string[]; set(v: string | string[], silent?: boolean): void }

/** type "single" keeps exactly one value on (radio-like roving tabindex); "multiple" toggles independently. */
export function toggleGroup(items: ToggleItem[], p: { type?: "single" | "multiple"; value?: string | string[]; variant?: "default" | "outline"; size?: "default" | "sm" | "lg"; class?: string; ariaLabel?: string; onChange?: (v: string[]) => void } = {}): ToggleGroupCtl {
  const single = (p.type ?? "single") === "single";
  let vals = new Set(Array.isArray(p.value) ? p.value : p.value != null ? [p.value] : single && items[0] ? [items[0].value] : []);
  const outline = p.variant === "outline";
  const el = h("div", { "data-slot": "toggle-group", role: single ? "radiogroup" : "group", "aria-label": p.ariaLabel, class: cn("group/toggle-group flex w-fit items-center rounded-md", outline && "shadow-xs", p.class) });
  const btns = items.map((it) => {
    const b = h("button", {
      type: "button", "data-slot": "toggle-group-item", "data-value": it.value, title: it.title, "aria-label": it.ariaLabel, disabled: it.disabled, role: single ? "radio" : undefined,
      class: cn(toggleVariants({ variant: p.variant, size: p.size }), "min-w-0 flex-1 shrink-0 rounded-none shadow-none first:rounded-l-md last:rounded-r-md focus:z-10 focus-visible:z-10", outline && "border-l-0 first:border-l"),
    }, it.icon && icon(it.icon), it.label);
    b.addEventListener("click", () => {
      if (single) vals = new Set([it.value]);
      else vals.has(it.value) ? vals.delete(it.value) : vals.add(it.value);
      sync(); p.onChange?.([...vals]);
    });
    return b;
  });
  const sync = () => btns.forEach((b, i) => {
    const on = vals.has(items[i]!.value);
    b.dataset.state = on ? "on" : "off";
    b.setAttribute(single ? "aria-checked" : "aria-pressed", String(on));
    if (single) b.tabIndex = on ? 0 : -1;
  });
  el.append(...btns);
  if (single) el.addEventListener("keydown", (e) => {
    const i = btns.indexOf(document.activeElement as HTMLButtonElement);
    if (i < 0) return;
    const n = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (n) { e.preventDefault(); const t = btns[(i + n + btns.length) % btns.length]!; t.focus(); t.click(); }
  });
  sync();
  return { el, get: () => [...vals], set: (v, silent = false) => { vals = new Set(Array.isArray(v) ? v : [v]); sync(); if (!silent) p.onChange?.([...vals]); } };
}
