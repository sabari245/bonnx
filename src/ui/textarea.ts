import { h } from "../lib/dom";
import { cn } from "../lib/cn";

export function textarea(p: { class?: string; value?: string; placeholder?: string; rows?: number; disabled?: boolean; ariaLabel?: string; onInput?: (v: string) => void } = {}): HTMLTextAreaElement {
  const el = h("textarea", {
    "data-slot": "textarea", rows: p.rows, placeholder: p.placeholder, disabled: p.disabled, "aria-label": p.ariaLabel, spellcheck: false,
    class: cn("border-input placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-ring/50 aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive dark:bg-input/30 flex field-sizing-content min-h-16 w-full rounded-md border bg-transparent px-3 py-2 text-base shadow-xs transition-[color,box-shadow] outline-none focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50 md:text-sm", p.class),
  });
  if (p.value != null) el.value = p.value;
  if (p.onInput) el.addEventListener("input", () => p.onInput!(el.value));
  return el;
}
