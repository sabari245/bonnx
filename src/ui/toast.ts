import { h, icon } from "../lib/dom";
import { cn } from "../lib/cn";
import { CircleAlert, CircleCheck, Info, X } from "lucide";
import { exit, portalRoot } from "./overlay";

export interface ToastOpts { title: string; description?: string | Node; variant?: "default" | "success" | "destructive"; duration?: number; action?: { label: string; onClick: () => void } }
let region: HTMLElement | null = null;

/** Sonner-style stacked toasts (bottom-right, newest on the bottom). Returns a dismiss function. duration 0 = sticky. */
export function toast(o: ToastOpts): () => void {
  if (!region || !region.isConnected) {
    region = h("div", { "aria-live": "polite", "aria-label": "Notifications", class: "pointer-events-none fixed right-4 bottom-4 z-[200] flex w-[min(24rem,calc(100vw-2rem))] flex-col gap-2" });
    portalRoot().appendChild(region);
  }
  const v = o.variant ?? "default";
  const ic = v === "success" ? icon(CircleCheck, "size-4 text-emerald-500 mt-0.5") : v === "destructive" ? icon(CircleAlert, "size-4 text-destructive mt-0.5") : icon(Info, "size-4 text-primary mt-0.5");
  let timer = 0;
  const dismiss = () => { clearTimeout(timer); exit(el); };
  const el = h("div", { role: v === "destructive" ? "alert" : "status", "data-state": "open", class: cn("ui-toast bg-popover text-popover-foreground pointer-events-auto relative flex items-start gap-3 rounded-lg border p-4 pr-9 shadow-lg", v === "destructive" && "border-destructive/40") },
    ic,
    h("div", { class: "grid min-w-0 flex-1 gap-1" },
      h("div", { class: "text-sm leading-tight font-medium" }, o.title),
      o.description && h("div", { class: "text-muted-foreground text-sm break-words" }, o.description),
      o.action && h("button", { type: "button", class: "text-primary mt-1 w-fit text-sm font-medium hover:underline", onclick: () => { o.action!.onClick(); dismiss(); } }, o.action.label)),
    h("button", { type: "button", "aria-label": "Dismiss", class: "absolute top-3 right-3 rounded-xs opacity-60 hover:opacity-100 focus-visible:ring-2 focus-visible:ring-ring outline-none", onclick: dismiss }, icon(X, "size-3.5")));
  region.appendChild(el);
  const d = o.duration ?? 4500;
  if (d > 0) {
    const start = () => (timer = window.setTimeout(dismiss, d));
    el.addEventListener("pointerenter", () => clearTimeout(timer));
    el.addEventListener("pointerleave", start);
    start();
  }
  return dismiss;
}
