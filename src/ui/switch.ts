import { h } from "../lib/dom";
import { cn } from "../lib/cn";

export interface SwitchCtl { el: HTMLButtonElement; get(): boolean; set(v: boolean, silent?: boolean): void }
export function switchControl(p: { checked?: boolean; disabled?: boolean; ariaLabel?: string; id?: string; class?: string; onChange?: (v: boolean) => void } = {}): SwitchCtl {
  let on = !!p.checked;
  const thumb = h("span", { "data-slot": "switch-thumb", class: "bg-background dark:data-[state=unchecked]:bg-foreground dark:data-[state=checked]:bg-primary-foreground pointer-events-none block size-4 rounded-full ring-0 transition-transform data-[state=checked]:translate-x-[calc(100%-2px)] data-[state=unchecked]:translate-x-0" });
  const el = h("button", {
    type: "button", role: "switch", id: p.id, disabled: p.disabled, "aria-label": p.ariaLabel, "data-slot": "switch",
    class: cn("peer data-[state=checked]:bg-primary data-[state=unchecked]:bg-input focus-visible:border-ring focus-visible:ring-ring/50 dark:data-[state=unchecked]:bg-input/80 inline-flex h-[1.15rem] w-8 shrink-0 items-center rounded-full border border-transparent shadow-xs transition-all outline-none focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50", p.class),
  }, thumb);
  const set = (v: boolean, silent = false) => {
    on = v;
    el.setAttribute("aria-checked", String(v));
    el.dataset.state = thumb.dataset.state = v ? "checked" : "unchecked";
    if (!silent) p.onChange?.(v);
  };
  el.addEventListener("click", () => set(!on));
  set(on, true);
  return { el, get: () => on, set };
}
