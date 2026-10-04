import { h } from "../lib/dom";
import { cn } from "../lib/cn";

export interface InputProps {
  class?: string;
  type?: string;
  value?: string;
  placeholder?: string;
  disabled?: boolean;
  name?: string;
  id?: string;
  ariaLabel?: string;
  autocomplete?: string;
  onInput?: (v: string, e: Event) => void;
  onChange?: (v: string, e: Event) => void;
}

export const inputClass =
  "file:text-foreground placeholder:text-muted-foreground selection:bg-primary selection:text-primary-foreground dark:bg-input/30 border-input h-9 w-full min-w-0 rounded-md border bg-transparent px-3 py-1 text-base shadow-xs transition-[color,box-shadow] outline-none file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-sm file:font-medium disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50 md:text-sm focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive";

export function input(p: InputProps = {}): HTMLInputElement {
  const el = h("input", {
    "data-slot": "input", type: p.type ?? "text", class: cn(inputClass, p.class), placeholder: p.placeholder, disabled: p.disabled,
    name: p.name, id: p.id, "aria-label": p.ariaLabel, autocomplete: p.autocomplete ?? "off", spellcheck: false,
  });
  if (p.value != null) el.value = p.value;
  if (p.onInput) el.addEventListener("input", (e) => p.onInput!(el.value, e));
  if (p.onChange) el.addEventListener("change", (e) => p.onChange!(el.value, e));
  return el;
}
