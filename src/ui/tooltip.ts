import { h } from "../lib/dom";
import { portalRoot, position, type Side, type Align } from "./overlay";

let tip: HTMLElement | null = null, showT = 0, owner: Element | null = null;
const TIP_CLASS = "ui-pop bg-foreground text-background pointer-events-none fixed z-[100] w-fit max-w-xs rounded-md px-3 py-1.5 text-xs text-balance shadow-md";

function hide(): void {
  clearTimeout(showT);
  if (tip) { tip.remove(); tip = null; }
  owner?.removeAttribute("aria-describedby");
  owner = null;
}

export interface TooltipOpts { side?: Side; align?: Align; delay?: number; sideOffset?: number }

/** Attach a delayed tooltip to `el`. `text` may be a function for dynamic content. Returns a detach function. */
export function tooltip(el: HTMLElement, text: string | (() => string), o: TooltipOpts = {}): () => void {
  const open = () => {
    const t = typeof text === "function" ? text() : text;
    if (!t) return;
    hide();
    tip = h("div", { role: "tooltip", id: "ui-tooltip", "data-state": "open", class: TIP_CLASS }, t);
    portalRoot().appendChild(tip);
    position(tip, el, { side: o.side ?? "top", align: o.align ?? "center", sideOffset: o.sideOffset ?? 6 });
    owner = el; el.setAttribute("aria-describedby", "ui-tooltip");
  };
  const enter = () => { clearTimeout(showT); showT = window.setTimeout(open, o.delay ?? 450); };
  const focus = () => { if (el.matches(":focus-visible")) { clearTimeout(showT); open(); } };
  el.addEventListener("pointerenter", enter);
  el.addEventListener("pointerleave", hide);
  el.addEventListener("pointerdown", hide);
  el.addEventListener("focus", focus);
  el.addEventListener("blur", hide);
  el.addEventListener("keydown", (e) => e.key === "Escape" && hide());
  return () => { hide(); el.removeEventListener("pointerenter", enter); el.removeEventListener("pointerleave", hide); el.removeEventListener("pointerdown", hide); el.removeEventListener("focus", focus); el.removeEventListener("blur", hide); };
}
export const hideTooltip = hide;
