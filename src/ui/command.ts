import { h, icon } from "../lib/dom";
import { cn } from "../lib/cn";
import { Search, type IconNode } from "lucide";
import { dialog, type DialogCtl } from "./dialog";
import { nextId } from "./overlay";

export interface CommandItem {
  id: string;
  label: string;
  /** right-aligned secondary text (e.g. shape, count) */
  hint?: string;
  group?: string;
  icon?: IconNode;
  /** small color chip drawn before the label */
  swatch?: string;
  keywords?: string;
  shortcut?: string;
  disabled?: boolean;
  onSelect?: (item: CommandItem) => void;
}
export interface CommandOpts {
  items?: CommandItem[];
  placeholder?: string;
  empty?: string;
  /** false → no internal filtering; the caller supplies already-filtered items via onQuery/setItems */
  filter?: boolean;
  maxItems?: number;
  class?: string;
  /** called (debounced 0ms) on every input change */
  onQuery?: (q: string) => void;
  onSelect?: (item: CommandItem) => void;
  /** called when the highlighted row changes (preview) */
  onActive?: (item: CommandItem | null) => void;
  /** clear the input after selecting */
  clearOnSelect?: boolean;
}
export interface CommandCtl { el: HTMLElement; input: HTMLInputElement; setItems(items: CommandItem[]): void; setQuery(q: string): void; focus(): void; getActive(): CommandItem | null }

/** fuzzy subsequence score: higher is better, -1 = no match. Prefers prefix/word-start/contiguous matches. */
export function fuzzy(q: string, text: string): number {
  q = q.toLowerCase(); const t = text.toLowerCase();
  if (!q) return 0;
  const idx = t.indexOf(q);
  if (idx >= 0) return 1000 - idx - (t.length - q.length) * 0.1 + (idx === 0 ? 200 : /[\s_./-]/.test(t[idx - 1] ?? "") ? 100 : 0);
  let ti = 0, score = 0, run = 0;
  for (const ch of q) {
    const f = t.indexOf(ch, ti);
    if (f < 0) return -1;
    run = f === ti ? run + 1 : 0;
    score += 10 + run * 5 + (f === 0 || /[\s_./-]/.test(t[f - 1] ?? "") ? 15 : 0) - (f - ti);
    ti = f + 1;
  }
  return Math.max(1, score);
}

/** Inline command list (search input + grouped, keyboard-navigable results). Use `commandDialog` for the Cmd-K overlay. */
export function command(o: CommandOpts = {}): CommandCtl {
  let all = o.items ?? [], shown: CommandItem[] = [], active = -1;
  const lid = nextId("cmd");
  const input = h("input", { type: "text", role: "combobox", "aria-expanded": "true", "aria-controls": lid, "aria-autocomplete": "list", autocomplete: "off", placeholder: o.placeholder ?? "Type to search…", "data-slot": "command-input", class: "placeholder:text-muted-foreground flex h-10 w-full rounded-md bg-transparent py-3 text-sm outline-hidden disabled:cursor-not-allowed disabled:opacity-50" });
  const list = h("div", { id: lid, role: "listbox", "data-slot": "command-list", class: "ui-scroll max-h-[min(24rem,60vh)] scroll-py-1 overflow-x-hidden overflow-y-auto p-1" });
  const el = h("div", { "data-slot": "command", class: cn("bg-popover text-popover-foreground flex h-full w-full flex-col overflow-hidden rounded-md", o.class) },
    h("div", { "data-slot": "command-input-wrapper", class: "flex h-10 items-center gap-2 border-b px-3" }, icon(Search, "size-4 shrink-0 opacity-50"), input), list);

  const rowFor = (it: CommandItem, i: number) => {
    const r = h("div", { role: "option", id: `${lid}-${i}`, "aria-selected": "false", "data-disabled": it.disabled ? "" : undefined, "data-i": i,
      class: "data-[selected=true]:bg-accent data-[selected=true]:text-accent-foreground [&_svg:not([class*='text-'])]:text-muted-foreground relative flex cursor-default items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-hidden select-none data-[disabled]:pointer-events-none data-[disabled]:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4" },
      it.swatch && h("i", { class: "size-2.5 shrink-0 rounded-[3px]", style: { background: it.swatch } }),
      it.icon && icon(it.icon),
      h("span", { class: "min-w-0 flex-1 truncate", title: it.label, dir: "ltr" }, it.label),
      it.hint && h("span", { class: "text-muted-foreground shrink-0 text-xs tabular-nums" }, it.hint),
      it.shortcut && h("span", { class: "text-muted-foreground ml-auto text-xs tracking-widest" }, it.shortcut));
    r.addEventListener("pointermove", () => { if (active !== i) setActive(i, false); });
    r.addEventListener("click", () => choose(it));
    return r;
  };
  const render = () => {
    list.replaceChildren();
    if (!shown.length) { list.append(h("div", { class: "text-muted-foreground py-6 text-center text-sm" }, o.empty ?? "No results.")); active = -1; o.onActive?.(null); return; }
    let lastGroup: string | undefined | null = null, grp: HTMLElement = list;
    shown.forEach((it, i) => {
      if (it.group !== lastGroup) {
        lastGroup = it.group;
        grp = it.group ? h("div", { role: "group", class: "text-foreground" }, h("div", { class: "text-muted-foreground px-2 py-1.5 text-xs font-medium" }, it.group)) : list;
        if (grp !== list) list.append(grp);
      }
      grp.append(rowFor(it, i));
    });
    setActive(shown.findIndex((x) => !x.disabled), true);
  };
  const setActive = (i: number, scroll: boolean) => {
    list.querySelector('[data-selected="true"]')?.removeAttribute("data-selected");
    list.querySelector('[aria-selected="true"]')?.setAttribute("aria-selected", "false");
    active = i;
    const r = list.querySelector<HTMLElement>(`[data-i="${i}"]`);
    if (r) { r.dataset.selected = "true"; r.setAttribute("aria-selected", "true"); input.setAttribute("aria-activedescendant", r.id); if (scroll) r.scrollIntoView({ block: "nearest" }); }
    o.onActive?.(shown[i] ?? null);
  };
  const choose = (it: CommandItem) => { if (it.disabled) return; it.onSelect?.(it); o.onSelect?.(it); if (o.clearOnSelect) { input.value = ""; refilter(); } };
  const refilter = () => {
    const q = input.value.trim(), max = o.maxItems ?? 200;
    if (o.filter === false || !q) shown = all.slice(0, max);
    else shown = all.map((it) => ({ it, s: Math.max(fuzzy(q, it.label), it.keywords ? fuzzy(q, it.keywords) - 50 : -1) })).filter((x) => x.s >= 0).sort((a, b) => b.s - a.s).slice(0, max).map((x) => x.it);
    render();
  };
  const move = (d: number) => {
    const n = shown.length; if (!n) return;
    let i = active;
    for (let k = 0; k < n; k++) { i = (i + d + n) % n; if (!shown[i]!.disabled) break; }
    setActive(i, true);
  };
  input.addEventListener("input", () => { o.onQuery?.(input.value); refilter(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
    else if (e.key === "Home" && e.ctrlKey) { e.preventDefault(); setActive(0, true); }
    else if (e.key === "Enter") { e.preventDefault(); const it = shown[active]; if (it) choose(it); }
  });
  refilter();
  return {
    el, input, focus: () => input.focus(),
    setItems(items) { all = items; refilter(); },
    setQuery(q) { input.value = q; refilter(); },
    getActive: () => shown[active] ?? null,
  };
}

/** Cmd/Ctrl-K style palette in a dialog. Returns the controller plus the command instance. */
export function commandDialog(o: CommandOpts & { onClose?: () => void }): { dialog: DialogCtl; command: CommandCtl; open(): void; close(): void } {
  const cmd = command({ ...o, onSelect: (it) => { dlg.close(); o.onSelect?.(it); } });
  const dlg = dialog({ size: "md", hideClose: true, class: "mt-[15vh] self-start p-0 gap-0 overflow-hidden", content: cmd.el, onClose: o.onClose });
  dlg.body.className = "min-h-0";
  return { dialog: dlg, command: cmd, open: () => { dlg.open(); cmd.setQuery(""); cmd.focus(); }, close: () => dlg.close() };
}
