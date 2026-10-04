import { h, icon } from "../lib/dom";
import { cn } from "../lib/cn";
import { Check, ChevronRight, Circle, type IconNode } from "lucide";
import { exit, portalRoot, position, pushLayer, type Align, type Side } from "./overlay";

/** Item model shared by dropdown-menu and context-menu. */
export type MenuItem =
  | { type?: "item"; label: string; icon?: IconNode; shortcut?: string; disabled?: boolean; destructive?: boolean; onSelect?: () => void }
  | { type: "checkbox"; label: string; checked: boolean; shortcut?: string; disabled?: boolean; onChange?: (v: boolean) => void }
  | { type: "radio"; label: string; value: string; group: string; checked: boolean; disabled?: boolean; onSelect?: (value: string) => void }
  | { type: "separator" }
  | { type: "label"; label: string }
  | { type: "sub"; label: string; icon?: IconNode; items: MenuItem[]; disabled?: boolean };

export const menuContentClass = "ui-pop bg-popover text-popover-foreground fixed z-50 max-h-[var(--menu-max-h,70vh)] min-w-[8rem] overflow-x-hidden overflow-y-auto rounded-md border p-1 shadow-md outline-none";
const itemClass = "focus:bg-accent focus:text-accent-foreground data-[variant=destructive]:text-destructive data-[variant=destructive]:focus:bg-destructive/10 dark:data-[variant=destructive]:focus:bg-destructive/20 data-[variant=destructive]:focus:text-destructive [&_svg:not([class*='text-'])]:text-muted-foreground relative flex cursor-default items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-hidden select-none data-[disabled]:pointer-events-none data-[disabled]:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4 data-[state=open]:bg-accent";

export interface MenuHandle { el: HTMLElement; close(): void }

/** Build + show a menu at an anchor. Used by dropdownMenu / contextMenu / anything else. */
export function showMenu(items: MenuItem[], anchor: Element | { x: number; y: number }, o: { side?: Side; align?: Align; sideOffset?: number; ignore?: (t: Node) => boolean; onClose?: () => void; class?: string } = {}): MenuHandle {
  const subs: HTMLElement[] = [];
  let unreg = () => {};
  let closed = false;
  const root = build(items, closeAll, 0);
  root.className = cn(menuContentClass, o.class);
  root.dataset.state = "open";
  portalRoot().appendChild(root);
  position(root, anchor, { side: o.side ?? "bottom", align: o.align ?? "start", sideOffset: o.sideOffset ?? 4 });
  unreg = pushLayer(root, closeAll, (t) => !!o.ignore?.(t) || subs.some((s) => s.contains(t)));
  focusItem(root, 0);

  function closeAll() {
    if (closed) return;
    closed = true; unreg();
    subs.splice(0).forEach((s) => s.remove());
    exit(root); o.onClose?.();
  }

  function build(list: MenuItem[], close: () => void, depth: number): HTMLElement {
    const el = h("div", { role: "menu", tabindex: -1, "data-slot": "menu-content" });
    el.addEventListener("keydown", (e) => keys(e, el, depth));
    for (const it of list) {
      if (it.type === "separator") { el.append(h("div", { role: "separator", class: "bg-border -mx-1 my-1 h-px" })); continue; }
      if (it.type === "label") { el.append(h("div", { class: "px-2 py-1.5 text-sm font-medium" }, it.label)); continue; }
      const dis = "disabled" in it && it.disabled;
      const base: Record<string, unknown> = { tabindex: -1, "data-disabled": dis ? "" : undefined, "aria-disabled": dis ? "true" : undefined, class: itemClass };
      if (it.type === "checkbox" || it.type === "radio") {
        const on = it.checked;
        const row = h("div", { ...base, role: it.type === "radio" ? "menuitemradio" : "menuitemcheckbox", "aria-checked": String(on), class: cn(itemClass, "py-1.5 pr-2 pl-8") },
          h("span", { class: "pointer-events-none absolute left-2 flex size-3.5 items-center justify-center" }, on && icon(it.type === "radio" ? Circle : Check, it.type === "radio" ? "size-2 fill-current" : "size-4")),
          it.label, "shortcut" in it && it.shortcut && h("span", { class: "text-muted-foreground ml-auto text-xs tracking-widest" }, it.shortcut));
        row.addEventListener("click", () => { if (dis) return; it.type === "radio" ? it.onSelect?.(it.value) : it.onChange?.(!it.checked); close(); });
        el.append(row); continue;
      }
      if (it.type === "sub") {
        const row = h("div", { ...base, role: "menuitem", "aria-haspopup": "menu", "aria-expanded": "false" }, it.icon && icon(it.icon), it.label, icon(ChevronRight, "ml-auto size-4"));
        let subEl: HTMLElement | null = null, t = 0;
        const openSub = (focus: boolean) => {
          if (dis || subEl) return;
          subEl = build(it.items, close, depth + 1);
          subEl.className = cn(menuContentClass, "z-[60]"); subEl.dataset.state = "open";
          portalRoot().appendChild(subEl); subs.push(subEl);
          (subEl as unknown as { _up: () => void })._up = () => { closeSub(); row.focus(); };
          position(subEl, row, { side: "right", align: "start", sideOffset: 2, alignOffset: -4 });
          row.setAttribute("aria-expanded", "true"); row.dataset.state = "open";
          if (focus) focusItem(subEl, 0);
        };
        const closeSub = () => { if (!subEl) return; const s = subEl; subEl = null; subs.splice(subs.indexOf(s), 1); s.remove(); row.setAttribute("aria-expanded", "false"); delete row.dataset.state; };
        row.addEventListener("pointerenter", () => { clearTimeout(t); t = window.setTimeout(() => openSub(false), 120); });
        row.addEventListener("click", () => openSub(true));
        row.addEventListener("keydown", (e) => { if (e.key === "ArrowRight") { e.preventDefault(); e.stopPropagation(); openSub(true); } });
        el.addEventListener("pointerover", (e) => { if (!row.contains(e.target as Node) && !(subEl && subEl.contains(e.target as Node))) { clearTimeout(t); closeSub(); } });
        el.append(row); continue;
      }
      const row = h("div", { ...base, role: "menuitem", "data-variant": it.destructive ? "destructive" : undefined }, it.icon && icon(it.icon), it.label,
        it.shortcut && h("span", { class: "text-muted-foreground ml-auto text-xs tracking-widest" }, it.shortcut));
      row.addEventListener("click", () => { if (dis) return; close(); queueMicrotask(() => it.onSelect?.()); });
      el.append(row);
    }
    el.addEventListener("pointermove", (e) => { const r = (e.target as HTMLElement).closest<HTMLElement>('[role^="menuitem"]'); if (r && r !== document.activeElement && !r.hasAttribute("data-disabled")) r.focus({ preventScroll: true }); });
    return el;
  }

  function keys(e: KeyboardEvent, el: HTMLElement, depth: number) {
    const items = enabled(el);
    const i = items.indexOf(document.activeElement as HTMLElement);
    const go = (n: number) => { e.preventDefault(); e.stopPropagation(); items[(i + n + items.length) % items.length]?.focus(); };
    if (e.key === "ArrowDown") go(1);
    else if (e.key === "ArrowUp") go(i < 0 ? -1 : -1);
    else if (e.key === "Home") { e.preventDefault(); items[0]?.focus(); }
    else if (e.key === "End") { e.preventDefault(); items[items.length - 1]?.focus(); }
    else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); (document.activeElement as HTMLElement)?.click(); }
    else if (e.key === "ArrowLeft" && depth > 0) { e.preventDefault(); e.stopPropagation(); (el as unknown as { _up?: () => void })._up?.(); }
    else if (e.key === "Tab") { e.preventDefault(); closeAll(); }
    else if (e.key.length === 1 && !e.ctrlKey && !e.metaKey) { const m = items.find((x, j) => j > i && x.textContent!.trim().toLowerCase().startsWith(e.key.toLowerCase())) ?? items.find((x) => x.textContent!.trim().toLowerCase().startsWith(e.key.toLowerCase())); m?.focus(); }
  }
  return { el: root, close: closeAll };
}

const enabled = (el: HTMLElement) => [...el.children].filter((c): c is HTMLElement => (c as HTMLElement).getAttribute?.("role")?.startsWith("menuitem") === true && !(c as HTMLElement).hasAttribute("data-disabled"));
function focusItem(el: HTMLElement, i: number) { (enabled(el)[i] ?? el).focus({ preventScroll: true }); }
