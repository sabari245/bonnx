import { h } from "../lib/dom";
import { cn } from "../lib/cn";

export function kbd(text: string, cls?: string): HTMLElement {
  return h("kbd", { "data-slot": "kbd", class: cn("bg-muted text-muted-foreground pointer-events-none inline-flex h-5 w-fit min-w-5 items-center justify-center gap-1 rounded-sm px-1 font-sans text-[11px] font-medium select-none border border-b-2", cls) }, text);
}
/** key combo "Ctrl+K" → <kbd>Ctrl</kbd><kbd>K</kbd> */
export function kbdGroup(combo: string, cls?: string): HTMLElement {
  return h("span", { class: cn("inline-flex items-center gap-1", cls) }, combo.split("+").map((k) => kbd(k.trim())));
}
