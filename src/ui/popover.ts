import { h } from "../lib/dom";
import { cn } from "../lib/cn";
import { exit, nextId, portalRoot, position, pushLayer, type Align, type Side } from "./overlay";

export interface PopoverCtl { open(): void; close(): void; toggle(): void; isOpen(): boolean; el: HTMLElement | null }
export interface PopoverOpts { side?: Side; align?: Align; sideOffset?: number; class?: string; matchWidth?: boolean; onOpenChange?: (o: boolean) => void; role?: string }

/** Click-to-toggle popover anchored to `trigger`. `content` can be built lazily on each open. */
export function popover(trigger: HTMLElement, content: Node | (() => Node), o: PopoverOpts = {}): PopoverCtl {
  let el: HTMLElement | null = null, unreg = () => {};
  const id = nextId("pop");
  trigger.setAttribute("aria-haspopup", "dialog"); trigger.setAttribute("aria-expanded", "false"); trigger.setAttribute("aria-controls", id);
  const close = () => {
    if (!el) return;
    const e = el; el = null; unreg(); trigger.setAttribute("aria-expanded", "false"); exit(e); o.onOpenChange?.(false);
    if (document.activeElement === document.body || e.contains(document.activeElement)) trigger.focus({ preventScroll: true });
  };
  const open = () => {
    if (el) return;
    el = h("div", { id, role: o.role ?? "dialog", tabindex: -1, "data-slot": "popover-content", "data-state": "open", class: cn("ui-pop bg-popover text-popover-foreground fixed z-50 w-72 rounded-md border p-4 shadow-md outline-hidden", o.class) }, typeof content === "function" ? content() : content);
    portalRoot().appendChild(el);
    position(el, trigger, { side: o.side, align: o.align ?? "center", sideOffset: o.sideOffset, matchWidth: o.matchWidth });
    unreg = pushLayer(el, close, (t) => trigger.contains(t));
    trigger.setAttribute("aria-expanded", "true"); o.onOpenChange?.(true);
    (el.querySelector<HTMLElement>("input,button,[tabindex]") ?? el).focus({ preventScroll: true });
  };
  trigger.addEventListener("click", () => (el ? close() : open()));
  return { open, close, toggle: () => (el ? close() : open()), isOpen: () => !!el, get el() { return el; } };
}
