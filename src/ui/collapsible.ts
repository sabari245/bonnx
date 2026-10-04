import { h, icon } from "../lib/dom";
import { cn } from "../lib/cn";
import { ChevronDown } from "lucide";

export interface CollapsibleCtl { el: HTMLElement; trigger: HTMLElement; body: HTMLElement; get(): boolean; set(open: boolean): void }

/** Single collapsible section. `trigger` is a node (a default chevron button is built from `title` when omitted). */
export function collapsible(p: { title?: string | Node; trigger?: HTMLElement; open?: boolean; class?: string; contentClass?: string; onChange?: (o: boolean) => void }, ...content: (string | Node)[]): CollapsibleCtl {
  let open = !!p.open;
  const trigger = p.trigger ?? h("button", { type: "button", class: "flex w-full items-center justify-between gap-2 py-2 text-sm font-medium outline-none hover:underline focus-visible:ring-ring/50 focus-visible:ring-[3px] rounded-md [&[aria-expanded=true]>svg]:rotate-180" }, p.title, icon(ChevronDown, "size-4 text-muted-foreground transition-transform duration-200"));
  const inner = h("div", { class: p.contentClass }, ...content);
  const body = h("div", { "data-slot": "collapsible-content", class: "ui-collapsible" }, h("div", null, inner));
  const el = h("div", { "data-slot": "collapsible", class: p.class }, trigger, body);
  const set = (o: boolean) => {
    open = o; body.dataset.state = o ? "open" : "closed"; trigger.setAttribute("aria-expanded", String(o));
    body.toggleAttribute("inert", !o); body.setAttribute("aria-hidden", String(!o)); p.onChange?.(o);
  };
  trigger.addEventListener("click", () => set(!open));
  body.dataset.state = open ? "open" : "closed"; trigger.setAttribute("aria-expanded", String(open)); body.toggleAttribute("inert", !open);
  return { el, trigger, body: inner, get: () => open, set };
}

/** Accordion = stack of collapsibles with separators; type single closes siblings. */
export function accordion(items: { value: string; title: string | Node; content: Node | string; open?: boolean }[], p: { type?: "single" | "multiple"; class?: string } = {}): HTMLElement {
  const ctls: CollapsibleCtl[] = [];
  const el = h("div", { "data-slot": "accordion", class: cn(p.class) });
  for (const it of items) {
    const c = collapsible({
      open: it.open, class: "border-b last:border-b-0",
      trigger: h("button", { type: "button", class: "flex flex-1 w-full items-start justify-between gap-4 rounded-md py-4 text-left text-sm font-medium transition-all outline-none hover:underline focus-visible:ring-ring/50 focus-visible:ring-[3px] [&[aria-expanded=true]>svg]:rotate-180" }, it.title, icon(ChevronDown, "size-4 shrink-0 translate-y-0.5 text-muted-foreground transition-transform duration-200")),
      contentClass: "pb-4 text-sm",
      onChange: (o) => { if (o && p.type !== "multiple") ctls.forEach((x) => x !== c && x.get() && x.set(false)); },
    }, it.content);
    ctls.push(c); el.append(c.el);
  }
  return el;
}
