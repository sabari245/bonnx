import { cn } from "./cn";
import { createElement as lucideCreate, type IconNode } from "lucide";

type Child = Node | string | number | false | null | undefined | Child[];
type Props = {
  class?: string | false | null | undefined | (string | false | null | undefined)[];
  style?: string | Partial<CSSStyleDeclaration>;
  dataset?: Record<string, string | number | boolean | undefined>;
  [k: string]: unknown;
};

/**
 * Tiny hyperscript helper used by every UI component (no framework).
 *   h("div", { class: "flex gap-2", onclick: fn }, child, "text")
 * `on*` props are event listeners; `class` goes through cn() so tailwind-merge dedupes conflicts.
 */
export function h<K extends keyof HTMLElementTagNameMap>(tag: K, props?: Props | null, ...children: Child[]): HTMLElementTagNameMap[K];
export function h(tag: string, props?: Props | null, ...children: Child[]): HTMLElement;
export function h(tag: string, props?: Props | null, ...children: Child[]): HTMLElement {
  const el = document.createElement(tag);
  if (props) {
    for (const [k, v] of Object.entries(props)) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = cn(v as string);
      else if (k === "style") typeof v === "string" ? (el.style.cssText = v) : Object.assign(el.style, v);
      else if (k === "dataset") for (const [dk, dv] of Object.entries(v as object)) dv !== undefined && (el.dataset[dk] = String(dv));
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v as EventListener);
      else if (k in el && k !== "list" && typeof v !== "object") (el as unknown as Record<string, unknown>)[k] = v;
      else el.setAttribute(k, v === true ? "" : String(v));
    }
  }
  append(el, children);
  return el;
}

export function append(el: Node, children: Child[]): void {
  for (const c of children) {
    if (c == null || c === false) continue;
    if (Array.isArray(c)) append(el, c);
    else el.appendChild(typeof c === "object" ? c : document.createTextNode(String(c)));
  }
}

export function clear(el: Node): void {
  while (el.firstChild) el.removeChild(el.firstChild);
}

export function icon(node: IconNode, cls = "size-4"): SVGElement {
  return lucideCreate(node, { class: cn("shrink-0", cls), "aria-hidden": "true" }) as unknown as SVGElement;
}

/** escape for the rare innerHTML use */
export const esc = (s: unknown): string =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
