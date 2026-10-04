import { h } from "../lib/dom";
import { cn } from "../lib/cn";
/** Native scroll container with thin styled scrollbars (see .ui-scroll in globals.css). */
export const scrollArea = (cls?: string, ...children: (string | Node)[]) =>
  h("div", { "data-slot": "scroll-area", class: cn("ui-scroll relative", cls) }, ...children);
