import { cva, type VariantProps } from "class-variance-authority";
import { h, icon } from "../lib/dom";
import { cn } from "../lib/cn";
import type { IconNode } from "lucide";

export const toggleVariants = cva(
  "inline-flex items-center justify-center gap-2 rounded-md text-sm font-medium hover:bg-muted hover:text-muted-foreground disabled:pointer-events-none disabled:opacity-50 data-[state=on]:bg-accent data-[state=on]:text-accent-foreground [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 [&_svg]:shrink-0 focus-visible:border-ring focus-visible:ring-ring/50 outline-none focus-visible:ring-[3px] transition-[color,box-shadow] whitespace-nowrap",
  {
    variants: {
      variant: { default: "bg-transparent", outline: "border border-input bg-transparent shadow-xs hover:bg-accent hover:text-accent-foreground" },
      size: { default: "h-9 px-2 min-w-9", sm: "h-8 px-1.5 min-w-8", lg: "h-10 px-2.5 min-w-10" },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);
export type ToggleProps = VariantProps<typeof toggleVariants> & { icon?: IconNode; pressed?: boolean; class?: string; title?: string; ariaLabel?: string; disabled?: boolean; onChange?: (on: boolean) => void };
export interface ToggleCtl { el: HTMLButtonElement; get(): boolean; set(v: boolean, silent?: boolean): void }

export function toggle(label: string | Node | null, p: ToggleProps = {}): ToggleCtl {
  let on = !!p.pressed;
  const el = h("button", { type: "button", "data-slot": "toggle", title: p.title, "aria-label": p.ariaLabel, disabled: p.disabled, class: cn(toggleVariants({ variant: p.variant, size: p.size }), p.class) }, p.icon && icon(p.icon), label);
  const set = (v: boolean, silent = false) => { on = v; el.setAttribute("aria-pressed", String(v)); el.dataset.state = v ? "on" : "off"; if (!silent) p.onChange?.(v); };
  el.addEventListener("click", () => set(!on));
  set(on, true);
  return { el, get: () => on, set };
}
