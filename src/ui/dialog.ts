import { h, icon } from "../lib/dom";
import { cn } from "../lib/cn";
import { X } from "lucide";
import { exit, nextId, portalRoot, pushLayer, trapFocus } from "./overlay";

export interface DialogOpts {
  title?: string | Node;
  description?: string | Node;
  content?: Node | string;
  footer?: Node | Node[];
  size?: "sm" | "md" | "lg" | "xl" | "full";
  /** hide the corner close button */
  hideClose?: boolean;
  class?: string;
  /** do not dismiss on outside click / Escape */
  modalOnly?: boolean;
  onClose?: () => void;
}
export interface DialogCtl { open(): void; close(): void; isOpen(): boolean; body: HTMLElement; content: HTMLElement; setTitle(t: string | Node): void; setDescription(t: string | Node): void }

const SIZES = { sm: "sm:max-w-sm", md: "sm:max-w-lg", lg: "sm:max-w-3xl", xl: "sm:max-w-6xl", full: "sm:max-w-[calc(100vw-2rem)] h-[calc(100vh-2rem)]" };
let locks = 0;
const lockScroll = (on: boolean) => { locks += on ? 1 : -1; document.documentElement.style.overflow = locks > 0 ? "hidden" : ""; };

/** Modal dialog (portal + overlay, focus trap, scroll lock). Build once, open/close many times; `body` persists. */
export function dialog(o: DialogOpts = {}): DialogCtl {
  const tid = nextId("dlg-t"), did = nextId("dlg-d");
  const titleEl = h("h2", { id: tid, "data-slot": "dialog-title", class: "text-lg leading-none font-semibold" }, o.title);
  const descEl = h("p", { id: did, "data-slot": "dialog-description", class: "text-muted-foreground text-sm" }, o.description);
  const body = h("div", { class: "min-h-0 flex-1 overflow-auto ui-scroll" }, o.content);
  const header = h("div", { "data-slot": "dialog-header", class: "flex flex-col gap-2 text-center sm:text-left" }, titleEl, descEl);
  const footer = o.footer ? h("div", { "data-slot": "dialog-footer", class: "flex flex-col-reverse gap-2 sm:flex-row sm:justify-end" }, o.footer) : null;
  const closeBtn = o.hideClose ? null : h("button", { type: "button", "aria-label": "Close", class: "ring-offset-background focus:ring-ring absolute top-4 right-4 rounded-xs opacity-70 transition-opacity hover:opacity-100 focus:ring-2 focus:ring-offset-2 focus:outline-hidden disabled:pointer-events-none [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4", onclick: () => close() }, icon(X));
  const content = h("div", { role: "dialog", "aria-modal": "true", "aria-labelledby": tid, "aria-describedby": did, tabindex: -1, "data-slot": "dialog-content", "data-state": "open",
    class: cn("ui-pop bg-background relative z-50 grid max-h-[calc(100vh-2rem)] w-full max-w-[calc(100%-2rem)] grid-rows-[auto_minmax(0,1fr)_auto] gap-4 rounded-lg border p-6 shadow-lg duration-200 outline-none", SIZES[o.size ?? "md"], o.class) },
    header, body, footer, closeBtn);
  if (!o.title && !o.description) header.remove();
  else if (!o.description) descEl.remove();
  const overlay = h("div", { "data-slot": "dialog-overlay", "data-state": "open", class: "ui-overlay fixed inset-0 z-50 bg-black/50" });
  const wrap = h("div", { class: "fixed inset-0 z-50 flex items-center justify-center" }, overlay, content);
  let isOpen = false, release = () => {}, unreg = () => {};
  overlay.addEventListener("pointerdown", () => !o.modalOnly && close());
  const close = () => {
    if (!isOpen) return;
    isOpen = false; unreg(); release(); lockScroll(false);
    wrap.dataset.state = "closed"; overlay.dataset.state = "closed"; content.dataset.state = "closed";
    exit(content, () => wrap.remove()); o.onClose?.();
  };
  const open = () => {
    if (isOpen) return;
    isOpen = true; lockScroll(true);
    overlay.dataset.state = "open"; content.dataset.state = "open";
    portalRoot().appendChild(wrap);
    release = trapFocus(content);
    unreg = pushLayer(content, () => !o.modalOnly && close(), () => true);
    (content.querySelector<HTMLElement>("[autofocus]") ?? content).focus({ preventScroll: true });
  };
  return { open, close, isOpen: () => isOpen, body, content, setTitle: (t) => titleEl.replaceChildren(t as Node), setDescription: (t) => descEl.replaceChildren(t as Node) };
}

/** Confirm helper: resolves true on confirm. */
export function confirmDialog(title: string, description: string, confirmLabel = "Continue"): Promise<boolean> {
  return new Promise((res) => {
    let result = false;
    const d = dialog({ title, description, size: "sm", hideClose: true, onClose: () => res(result),
      footer: [
        h("button", { type: "button", class: "border bg-background shadow-xs hover:bg-accent hover:text-accent-foreground inline-flex h-9 items-center justify-center rounded-md px-4 text-sm font-medium", onclick: () => d.close() }, "Cancel"),
        h("button", { type: "button", autofocus: true, class: "bg-primary text-primary-foreground shadow-xs hover:bg-primary/90 inline-flex h-9 items-center justify-center rounded-md px-4 text-sm font-medium", onclick: () => { result = true; d.close(); } }, confirmLabel),
      ] });
    d.open();
  });
}
