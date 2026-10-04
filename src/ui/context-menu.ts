import { showMenu, type MenuHandle, type MenuItem } from "./menu";

/** Right-click (and Shift+F10 / ContextMenu key) menu on `target`. `items` may depend on the event. Returns a programmatic opener. */
export function contextMenu(target: HTMLElement, items: MenuItem[] | ((e: MouseEvent | KeyboardEvent) => MenuItem[] | null)): { openAt(x: number, y: number, list?: MenuItem[]): void; close(): void } {
  let cur: MenuHandle | null = null;
  const openAt = (x: number, y: number, list?: MenuItem[]) => {
    cur?.close();
    if (!list) return;
    cur = showMenu(list, { x, y }, { side: "bottom", align: "start", sideOffset: 2, onClose: () => (cur = null) });
  };
  target.addEventListener("contextmenu", (e) => {
    const l = typeof items === "function" ? items(e) : items;
    if (!l) return;
    e.preventDefault();
    openAt(e.clientX, e.clientY, l);
  });
  target.addEventListener("keydown", (e) => {
    if (e.key === "ContextMenu" || (e.shiftKey && e.key === "F10")) {
      const l = typeof items === "function" ? items(e) : items;
      if (!l) return;
      e.preventDefault();
      const r = target.getBoundingClientRect();
      openAt(r.left + 16, r.top + 16, l);
    }
  });
  return { openAt, close: () => cur?.close() };
}
