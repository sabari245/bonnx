import { h } from "../lib/dom";
import { cn } from "../lib/cn";

export interface VirtualListOpts {
  rowHeight: number;
  count: number;
  /** build the element for row i; it is absolutely positioned & sized by the list */
  render(i: number): HTMLElement;
  overscan?: number;
  class?: string;
  /** minimum content width (enables horizontal scroll), px */
  minWidth?: number;
}
export interface VirtualListCtl { el: HTMLElement; setCount(n: number): void; refresh(): void; scrollToIndex(i: number, align?: "start" | "center" | "nearest"): void; getRange(): [number, number] }

/** Fixed-row-height virtualized list: only rows within the viewport (+overscan) exist in the DOM. Handles millions of rows. */
export function createVirtualList(o: VirtualListOpts): VirtualListCtl {
  let count = o.count;
  const over = o.overscan ?? 6;
  const rows = h("div", { class: "absolute top-0 left-0 w-full" });
  const sizer = h("div", { class: "relative w-full", style: { minWidth: o.minWidth ? `${o.minWidth}px` : "" } }, rows);
  const el = h("div", { "data-slot": "virtual-list", role: "list", class: cn("ui-scroll relative h-full overflow-auto", o.class) }, sizer);
  let a = -1, b = -1, raf = 0;
  const MAXPX = 16_000_000; // browsers cap element height; above this the scroll range is mapped proportionally
  const total = () => count * o.rowHeight;
  /** scrollTop → virtual offset into the full (unscaled) content */
  const vtop = () => {
    const T = total(), vh = el.clientHeight;
    return T <= MAXPX ? el.scrollTop : (el.scrollTop / Math.max(1, MAXPX - vh)) * (T - vh);
  };
  const sync = (force = false) => {
    const vt = vtop(), vh = el.clientHeight;
    const i0 = Math.max(0, Math.floor(vt / o.rowHeight) - over), i1 = Math.min(count, Math.ceil((vt + vh) / o.rowHeight) + over);
    if (force || i0 !== a || i1 !== b) {
      a = i0; b = i1;
      const frag = document.createDocumentFragment();
      for (let i = i0; i < i1; i++) {
        const r = o.render(i);
        r.style.position = "absolute"; r.style.top = `${(i - i0) * o.rowHeight}px`; r.style.height = `${o.rowHeight}px`; r.style.left = "0"; r.style.right = "0";
        r.setAttribute("role", "listitem"); r.dataset.index = String(i);
        frag.appendChild(r);
      }
      rows.replaceChildren(frag);
    }
    rows.style.top = `${el.scrollTop + a * o.rowHeight - vt}px`;
  };
  const resize = () => { sizer.style.height = `${Math.min(total(), MAXPX)}px`; };
  el.addEventListener("scroll", () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; sync(); }); }, { passive: true });
  new ResizeObserver(() => sync(true)).observe(el);
  resize();
  queueMicrotask(() => sync(true));
  return {
    el,
    setCount(n) { count = n; resize(); sync(true); },
    refresh() { sync(true); },
    scrollToIndex(i, align = "nearest") {
      const T = total(), vh = el.clientHeight, y = i * o.rowHeight;
      const toScroll = (v: number) => (T <= MAXPX ? v : (v / Math.max(1, T - vh)) * (MAXPX - vh));
      const cur = vtop();
      if (align === "start") el.scrollTop = toScroll(y);
      else if (align === "center") el.scrollTop = toScroll(y - vh / 2 + o.rowHeight / 2);
      else if (y < cur) el.scrollTop = toScroll(y);
      else if (y + o.rowHeight > cur + vh) el.scrollTop = toScroll(y + o.rowHeight - vh);
    },
    getRange: () => [a, b],
  };
}

export interface VirtualTableColumn { header: string | Node; width: number; align?: "left" | "right" | "center"; class?: string }
/** Header + virtualized body that scroll together horizontally. `cell(i, c)` returns text or node. */
export function createVirtualTable(o: { columns: VirtualTableColumn[]; rowHeight?: number; count: number; cell(row: number, col: number): string | Node; headerHeight?: number; class?: string; onRowClick?: (i: number) => void }): VirtualListCtl & { setColumns(c: VirtualTableColumn[]): void } {
  let cols = o.columns;
  const rh = o.rowHeight ?? 28, hh = o.headerHeight ?? 32;
  const totalW = () => cols.reduce((s, c) => s + c.width, 0);
  const head = h("div", { class: "bg-muted/60 text-muted-foreground sticky top-0 z-10 flex border-b text-xs font-medium", style: { height: `${hh}px` } });
  const mkHead = () => head.replaceChildren(...cols.map((c) => h("div", { class: cn("shrink-0 truncate px-2 leading-8", c.align === "right" && "text-right", c.align === "center" && "text-center", c.class), style: { width: `${c.width}px`, lineHeight: `${hh}px` } }, c.header)));
  mkHead();
  const list = createVirtualList({
    rowHeight: rh, count: o.count, minWidth: totalW(), class: "!overflow-visible",
    render: (i) => {
      const r = h("div", { class: "hover:bg-muted/50 flex border-b font-mono text-xs tabular-nums", style: { width: `${totalW()}px` }, onclick: o.onRowClick ? () => o.onRowClick!(i) : undefined },
        ...cols.map((c, ci) => h("div", { class: cn("shrink-0 truncate px-2", c.align === "right" && "text-right", c.align === "center" && "text-center", c.class), style: { width: `${c.width}px`, lineHeight: `${rh - 1}px` } }, o.cell(i, ci))));
      return r;
    },
  });
  // vertical/horizontal scrolling is done by an outer scroller so the sticky header tracks both axes
  const outer = h("div", { class: cn("ui-scroll relative h-full overflow-auto", o.class) }, head, list.el);
  list.el.style.height = "auto";
  // the inner list needs its own viewport metrics: delegate scroll to `outer`
  Object.defineProperty(list.el, "scrollTop", { get: () => Math.max(0, outer.scrollTop), set: (v: number) => (outer.scrollTop = v + 0) });
  Object.defineProperty(list.el, "clientHeight", { get: () => Math.max(0, outer.clientHeight - hh) });
  outer.addEventListener("scroll", () => list.el.dispatchEvent(new Event("scroll")), { passive: true });
  new ResizeObserver(() => list.refresh()).observe(outer);
  return { ...list, el: outer, setColumns(c) { cols = c; mkHead(); list.refresh(); } };
}
