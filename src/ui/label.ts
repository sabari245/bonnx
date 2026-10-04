import { h } from "../lib/dom";
import { cn } from "../lib/cn";

export function label(text: string | Node, p: { for?: string; class?: string } = {}): HTMLLabelElement {
  return h("label", { "data-slot": "label", htmlFor: p.for, class: cn("flex items-center gap-2 text-sm leading-none font-medium select-none group-data-[disabled=true]:pointer-events-none group-data-[disabled=true]:opacity-50 peer-disabled:cursor-not-allowed peer-disabled:opacity-50", p.class) }, text);
}
