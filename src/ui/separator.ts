import { h } from "../lib/dom";
import { cn } from "../lib/cn";

export function separator(orientation: "horizontal" | "vertical" = "horizontal", cls?: string): HTMLElement {
  return h("div", {
    "data-slot": "separator", role: "separator", "aria-orientation": orientation === "vertical" ? "vertical" : undefined,
    class: cn("bg-border shrink-0", orientation === "horizontal" ? "h-px w-full" : "h-full w-px self-stretch", cls),
  });
}
