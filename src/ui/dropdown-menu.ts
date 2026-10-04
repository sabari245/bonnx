import { showMenu, type MenuHandle, type MenuItem } from "./menu";
import type { Align, Side } from "./overlay";

export type { MenuItem } from "./menu";

/** Attach a dropdown menu to `trigger`. `items` may be a function evaluated at open time (for live checked state). */
export function dropdownMenu(trigger: HTMLElement, items: MenuItem[] | (() => MenuItem[]), o: { side?: Side; align?: Align; class?: string } = {}): { open(): void; close(): void } {
  let cur: MenuHandle | null = null;
  trigger.setAttribute("aria-haspopup", "menu"); trigger.setAttribute("aria-expanded", "false");
  const close = () => cur?.close();
  const open = () => {
    if (cur) return close();
    cur = showMenu(typeof items === "function" ? items() : items, trigger, {
      side: o.side ?? "bottom", align: o.align ?? "start", class: o.class, ignore: (t) => trigger.contains(t),
      onClose: () => { cur = null; trigger.setAttribute("aria-expanded", "false"); trigger.focus({ preventScroll: true }); },
    });
    trigger.setAttribute("aria-expanded", "true");
  };
  trigger.addEventListener("click", open);
  trigger.addEventListener("keydown", (e) => { if (e.key === "ArrowDown" && !cur) { e.preventDefault(); open(); } });
  return { open, close };
}
