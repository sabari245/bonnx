import { h } from "../lib/dom";
import { cn } from "../lib/cn";
export const skeleton = (cls?: string) => h("div", { "data-slot": "skeleton", class: cn("bg-accent ui-pulse rounded-md", cls) });
