import { h, icon } from "../lib/dom";
import { cn } from "../lib/cn";
import { X } from "lucide";
import { dragResize } from "./resizable";
import { exit, portalRoot, pushLayer, trapFocus } from "./overlay";

export interface SheetOpts {
  title?: string | Node;
  /** extra nodes placed in the header, right-aligned (e.g. action buttons) */
  actions?: Node[];
  width?: number;
  minWidth?: number;
  maxWidth?: number;
  /** modal = overlay + focus trap + Escape; non-modal = docked panel the user can interact beside (graph inspector) */
  modal?: boolean;
  /** mount point for non-modal panels (must be position:relative/absolute-friendly); default: fixed to viewport */
  mount?: HTMLElement;
  /** persisted width key */
  storageKey?: string;
  class?: string;
  onOpenChange?: (open: boolean) => void;
  onResize?: (w: number) => void;
}
export interface SheetCtl { el: HTMLElement; body: HTMLElement; open(): void; close(): void; isOpen(): boolean; setTitle(t: string | Node): void; setContent(...n: (Node | string)[]): void; getWidth(): number; setWidth(w: number): void }

/** Right-hand resizable side panel. */
export function sheet(o: SheetOpts = {}): SheetCtl {
  let width = o.width ?? 420;
  if (o.storageKey) try { const v = Number(localStorage.getItem(o.storageKey)); if (v > 0) width = v; } catch { /* ignore */ }
  const min = o.minWidth ?? 300, max = () => Math.min(o.maxWidth ?? 900, innerWidth - 40);
  const titleEl = h("h2", { "data-slot": "sheet-title", class: "text-foreground min-w-0 flex-1 truncate font-semibold" }, o.title);
  const closeBtn = h("button", { type: "button", "aria-label": "Close panel", class: "ring-offset-background focus:ring-ring rounded-xs p-1 opacity-70 transition-opacity hover:opacity-100 focus:ring-2 focus:ring-offset-2 focus:outline-hidden", onclick: () => close() }, icon(X));
  const header = h("div", { "data-slot": "sheet-header", class: "flex h-12 shrink-0 items-center gap-2 border-b px-4" }, titleEl, ...(o.actions ?? []), closeBtn);
  const body = h("div", { "data-slot": "sheet-body", class: "ui-scroll min-h-0 flex-1" });
  const grip = h("div", { "data-slot": "sheet-resize", "aria-label": "Resize panel", class: "hover:bg-primary/50 data-[dragging]:bg-primary absolute inset-y-0 -left-1 z-10 w-2 cursor-col-resize transition-colors" });
  const el = h("aside", { "data-slot": "sheet-content", "data-state": "closed", class: cn("ui-sheet bg-background flex flex-col border-l shadow-lg", o.mount ? "absolute inset-y-0 right-0 z-30" : "fixed inset-y-0 right-0 z-50", o.class), style: { width: `${width}px`, maxWidth: "100vw" } }, grip, header, body);
  el.setAttribute("aria-label", typeof o.title === "string" ? o.title : "Details");
  if (o.modal) { el.setAttribute("role", "dialog"); el.setAttribute("aria-modal", "true"); el.tabIndex = -1; }
  dragResize(grip, { axis: "x", invert: true, min, max, get: () => width, set: (v) => { width = v; el.style.width = `${v}px`; o.onResize?.(v); },
    onEnd: (v) => { if (o.storageKey) try { localStorage.setItem(o.storageKey, String(Math.round(v))); } catch { /* ignore */ } } });
  let isOpen = false, overlay: HTMLElement | null = null, release = () => {}, unreg = () => {};
  const open = () => {
    if (isOpen) { el.dataset.state = "open"; return; }
    isOpen = true; el.dataset.state = "open";
    if (o.modal) {
      overlay = h("div", { class: "ui-overlay fixed inset-0 z-50 bg-black/50", "data-state": "open", onpointerdown: () => close() });
      portalRoot().append(overlay, el);
      release = trapFocus(el); unreg = pushLayer(el, close, () => true); el.focus({ preventScroll: true });
    } else (o.mount ?? portalRoot()).appendChild(el);
    o.onOpenChange?.(true);
  };
  const close = () => {
    if (!isOpen) return;
    isOpen = false; unreg(); release();
    overlay?.remove(); overlay = null;
    exit(el); o.onOpenChange?.(false);
  };
  return {
    el, body, open, close, isOpen: () => isOpen,
    setTitle: (t) => { titleEl.replaceChildren(t as Node); el.setAttribute("aria-label", typeof t === "string" ? t : "Details"); },
    setContent: (...n) => { body.replaceChildren(...n); body.scrollTop = 0; },
    getWidth: () => width, setWidth: (w) => { width = Math.max(min, Math.min(max(), w)); el.style.width = `${width}px`; o.onResize?.(width); },
  };
}
