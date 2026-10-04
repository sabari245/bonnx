import { h } from "../lib/dom";
import { cn } from "../lib/cn";

export interface TabItem { value: string; label: string | Node; content: Node | (() => Node); disabled?: boolean }
export interface TabsCtl { el: HTMLElement; list: HTMLElement; get(): string; set(v: string, silent?: boolean): void; panel(v: string): HTMLElement }

/** Content given as a function is built lazily the first time its tab opens. */
export function tabs(items: TabItem[], p: { value?: string; class?: string; listClass?: string; contentClass?: string; onValueChange?: (v: string) => void } = {}): TabsCtl {
  let val = p.value ?? items[0]!.value;
  const list = h("div", { role: "tablist", "data-slot": "tabs-list", class: cn("bg-muted text-muted-foreground inline-flex h-9 w-fit items-center justify-center rounded-lg p-[3px]", p.listClass) });
  const el = h("div", { "data-slot": "tabs", class: cn("flex flex-col gap-2", p.class) }, list);
  const uid = Math.random().toString(36).slice(2, 7);
  const triggers = new Map<string, HTMLButtonElement>(), panels = new Map<string, HTMLElement>(), built = new Set<string>();
  for (const it of items) {
    const t = h("button", {
      type: "button", role: "tab", id: `${uid}-t-${it.value}`, "aria-controls": `${uid}-p-${it.value}`, disabled: it.disabled, "data-slot": "tabs-trigger",
      class: "data-[state=active]:bg-background dark:data-[state=active]:text-foreground focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:outline-ring dark:data-[state=active]:border-input dark:data-[state=active]:bg-input/30 text-foreground dark:text-muted-foreground inline-flex h-[calc(100%-1px)] flex-1 items-center justify-center gap-1.5 rounded-md border border-transparent px-2 py-1 text-sm font-medium whitespace-nowrap transition-[color,box-shadow] focus-visible:ring-[3px] focus-visible:outline-1 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:shadow-sm [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
      onclick: () => set(it.value),
    }, it.label);
    const pn = h("div", { role: "tabpanel", id: `${uid}-p-${it.value}`, "aria-labelledby": t.id, tabindex: 0, "data-slot": "tabs-content", class: cn("flex-1 outline-none", p.contentClass) });
    pn.hidden = true;
    triggers.set(it.value, t); panels.set(it.value, pn);
    list.append(t); el.append(pn);
  }
  list.addEventListener("keydown", (e) => {
    const en = items.filter((i) => !i.disabled);
    const i = en.findIndex((x) => x.value === val);
    const n = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (n) { e.preventDefault(); const nx = en[(i + n + en.length) % en.length]!; set(nx.value); triggers.get(nx.value)!.focus(); }
    else if (e.key === "Home") { e.preventDefault(); set(en[0]!.value); triggers.get(en[0]!.value)!.focus(); }
    else if (e.key === "End") { e.preventDefault(); const l = en[en.length - 1]!; set(l.value); triggers.get(l.value)!.focus(); }
  });
  function set(v: string, silent = false) {
    val = v;
    for (const it of items) {
      const on = it.value === v, t = triggers.get(it.value)!, pn = panels.get(it.value)!;
      t.dataset.state = on ? "active" : "inactive"; t.setAttribute("aria-selected", String(on)); t.tabIndex = on ? 0 : -1;
      pn.hidden = !on; pn.dataset.state = on ? "active" : "inactive";
      if (on && !built.has(it.value)) { built.add(it.value); pn.append(typeof it.content === "function" ? it.content() : it.content); }
    }
    if (!silent) p.onValueChange?.(v);
  }
  set(val, true);
  return { el, list, get: () => val, set, panel: (v) => panels.get(v)! };
}
