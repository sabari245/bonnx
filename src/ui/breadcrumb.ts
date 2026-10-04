import { h, icon } from "../lib/dom";
import { cn } from "../lib/cn";
import { ChevronRight } from "lucide";

export interface Crumb { label: string; onClick?: () => void; title?: string }
/** The last crumb is rendered as the current page. */
export function breadcrumb(items: Crumb[], cls?: string): HTMLElement {
  const ol = h("ol", { class: "text-muted-foreground flex flex-wrap items-center gap-1.5 text-sm break-words sm:gap-2.5" });
  items.forEach((c, i) => {
    const last = i === items.length - 1;
    ol.append(h("li", { "data-slot": "breadcrumb-item", class: "inline-flex items-center gap-1.5" },
      last ? h("span", { role: "link", "aria-disabled": "true", "aria-current": "page", title: c.title, class: "text-foreground font-normal max-w-[24ch] truncate" }, c.label)
        : h("button", { type: "button", title: c.title, class: "hover:text-foreground transition-colors max-w-[24ch] truncate rounded-sm outline-none focus-visible:ring-ring/50 focus-visible:ring-[3px]", onclick: c.onClick }, c.label)));
    if (!last) ol.append(h("li", { role: "presentation", "aria-hidden": "true", class: "[&>svg]:size-3.5" }, icon(ChevronRight)));
  });
  return h("nav", { "aria-label": "breadcrumb", "data-slot": "breadcrumb", class: cn(cls) }, ol);
}
