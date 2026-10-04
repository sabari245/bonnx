import { h, icon } from "../lib/dom";
import { cn } from "../lib/cn";
import { Check, Minus } from "lucide";

export interface CheckboxCtl { el: HTMLButtonElement; get(): boolean | "indeterminate"; set(v: boolean | "indeterminate", silent?: boolean): void }
export function checkbox(p: { checked?: boolean | "indeterminate"; disabled?: boolean; ariaLabel?: string; id?: string; class?: string; onChange?: (v: boolean) => void } = {}): CheckboxCtl {
  let v: boolean | "indeterminate" = p.checked ?? false;
  const ind = h("span", { "data-slot": "checkbox-indicator", class: "flex items-center justify-center text-current transition-none" });
  const el = h("button", {
    type: "button", role: "checkbox", id: p.id, disabled: p.disabled, "aria-label": p.ariaLabel, "data-slot": "checkbox",
    class: cn("peer border-input dark:bg-input/30 data-[state=checked]:bg-primary data-[state=indeterminate]:bg-primary data-[state=checked]:text-primary-foreground data-[state=indeterminate]:text-primary-foreground dark:data-[state=checked]:bg-primary dark:data-[state=indeterminate]:bg-primary data-[state=checked]:border-primary data-[state=indeterminate]:border-primary focus-visible:border-ring focus-visible:ring-ring/50 size-4 shrink-0 rounded-[4px] border shadow-xs transition-shadow outline-none focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50", p.class),
  }, ind);
  const set = (nv: boolean | "indeterminate", silent = false) => {
    v = nv;
    el.setAttribute("aria-checked", nv === "indeterminate" ? "mixed" : String(nv));
    el.dataset.state = nv === "indeterminate" ? "indeterminate" : nv ? "checked" : "unchecked";
    ind.replaceChildren(...(nv === "indeterminate" ? [icon(Minus, "size-3.5")] : nv ? [icon(Check, "size-3.5")] : []));
    if (!silent) p.onChange?.(nv === true);
  };
  el.addEventListener("click", () => set(v !== true));
  set(v, true);
  return { el, get: () => v, set };
}
