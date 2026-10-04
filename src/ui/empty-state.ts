import { h, icon } from "../lib/dom";
import { cn } from "../lib/cn";
import type { IconNode } from "lucide";

export function emptyState(p: { icon?: IconNode; title: string; description?: string | Node; action?: Node; class?: string }): HTMLElement {
  return h("div", { "data-slot": "empty", class: cn("flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed p-8 text-center", p.class) },
    p.icon && h("div", { class: "bg-muted text-muted-foreground flex size-10 items-center justify-center rounded-lg" }, icon(p.icon, "size-5")),
    h("div", { class: "text-sm font-medium" }, p.title),
    p.description && h("div", { class: "text-muted-foreground max-w-sm text-sm" }, p.description),
    p.action);
}
