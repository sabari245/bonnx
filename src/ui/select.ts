import { h, icon } from "../lib/dom";
import { cn } from "../lib/cn";
import { Check, ChevronDown } from "lucide";
import { exit, nextId, portalRoot, position, pushLayer } from "./overlay";

export type SelectOption = { value: string; label: string; disabled?: boolean } | { group: string; options: { value: string; label: string; disabled?: boolean }[] };
export interface SelectCtl { el: HTMLButtonElement; get(): string; set(v: string, silent?: boolean): void }

export function select(options: SelectOption[], p: { value?: string; placeholder?: string; size?: "default" | "sm"; class?: string; ariaLabel?: string; disabled?: boolean; onChange?: (v: string) => void } = {}): SelectCtl {
  const flat = options.flatMap((o) => ("group" in o ? o.options : [o]));
  let val = p.value ?? "";
  const valueEl = h("span", { class: "line-clamp-1 flex items-center gap-2 text-left" });
  const trig = h("button", {
    type: "button", role: "combobox", "aria-expanded": "false", "aria-haspopup": "listbox", "aria-label": p.ariaLabel, disabled: p.disabled, "data-slot": "select-trigger", "data-size": p.size ?? "default",
    class: cn("border-input data-[placeholder]:text-muted-foreground [&_svg:not([class*='text-'])]:text-muted-foreground focus-visible:border-ring focus-visible:ring-ring/50 dark:bg-input/30 dark:hover:bg-input/50 flex w-fit items-center justify-between gap-2 rounded-md border bg-transparent px-3 py-2 text-sm whitespace-nowrap shadow-xs transition-[color,box-shadow] outline-none focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50 data-[size=default]:h-9 data-[size=sm]:h-8 *:data-[slot=select-value]:line-clamp-1 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4", p.class),
  }, valueEl, icon(ChevronDown, "size-4 opacity-50"));
  valueEl.dataset.slot = "select-value";
  const sync = () => {
    const o = flat.find((x) => x.value === val);
    valueEl.textContent = o ? o.label : p.placeholder ?? "";
    if (o) trig.removeAttribute("data-placeholder"); else trig.setAttribute("data-placeholder", "");
  };
  const set = (v: string, silent = false) => { val = v; sync(); if (!silent) p.onChange?.(v); };

  let list: HTMLElement | null = null, unreg = () => {};
  const id = nextId("sel");
  const close = (refocus = true) => { if (!list) return; const l = list; list = null; unreg(); trig.setAttribute("aria-expanded", "false"); exit(l); if (refocus) trig.focus({ preventScroll: true }); };
  const open = () => {
    if (list || p.disabled) return;
    list = h("div", { id, role: "listbox", tabindex: -1, "data-slot": "select-content", "data-state": "open", class: "ui-pop bg-popover text-popover-foreground fixed z-50 max-h-72 min-w-[8rem] overflow-y-auto overflow-x-hidden rounded-md border p-1 shadow-md outline-none" });
    const opts: HTMLElement[] = [];
    const mk = (o: { value: string; label: string; disabled?: boolean }) => {
      const on = o.value === val;
      const row = h("div", { role: "option", id: `${id}-${o.value}`, "aria-selected": String(on), "data-value": o.value, "data-disabled": o.disabled ? "" : undefined, class: "focus:bg-accent focus:text-accent-foreground data-[active]:bg-accent data-[active]:text-accent-foreground relative flex w-full cursor-default items-center gap-2 rounded-sm py-1.5 pr-8 pl-2 text-sm outline-hidden select-none data-[disabled]:pointer-events-none data-[disabled]:opacity-50" },
        o.label, h("span", { class: "absolute right-2 flex size-3.5 items-center justify-center" }, on && icon(Check, "size-4")));
      row.addEventListener("click", () => { if (o.disabled) return; set(o.value); close(); });
      row.addEventListener("pointermove", () => activate(row));
      opts.push(row); return row;
    };
    for (const o of options) {
      if ("group" in o) list.append(h("div", { role: "group" }, h("div", { class: "text-muted-foreground px-2 py-1.5 text-xs" }, o.group), o.options.map(mk)));
      else list.append(mk(o));
    }
    const activate = (row: HTMLElement | undefined) => {
      opts.forEach((r) => r.removeAttribute("data-active"));
      if (!row) return; row.setAttribute("data-active", ""); list!.setAttribute("aria-activedescendant", row.id); row.scrollIntoView({ block: "nearest" });
    };
    portalRoot().appendChild(list);
    position(list, trig, { side: "bottom", align: "start", sideOffset: 4, matchWidth: true });
    unreg = pushLayer(list, () => close(), (t) => trig.contains(t));
    trig.setAttribute("aria-expanded", "true"); trig.setAttribute("aria-controls", id);
    const en = () => opts.filter((r) => !r.hasAttribute("data-disabled"));
    activate(opts.find((r) => r.dataset.value === val) ?? en()[0]);
    list.focus({ preventScroll: true });
    let typed = "", tt = 0;
    list.addEventListener("keydown", (e) => {
      const e2 = en(), i = e2.findIndex((r) => r.hasAttribute("data-active"));
      if (e.key === "ArrowDown") { e.preventDefault(); activate(e2[Math.min(e2.length - 1, i + 1)]); }
      else if (e.key === "ArrowUp") { e.preventDefault(); activate(e2[Math.max(0, i - 1)]); }
      else if (e.key === "Home") { e.preventDefault(); activate(e2[0]); }
      else if (e.key === "End") { e.preventDefault(); activate(e2[e2.length - 1]); }
      else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e2[i]?.click(); }
      else if (e.key === "Tab") { e.preventDefault(); close(); }
      else if (e.key.length === 1) { typed += e.key.toLowerCase(); clearTimeout(tt); tt = window.setTimeout(() => (typed = ""), 600); activate(e2.find((r) => r.textContent!.toLowerCase().startsWith(typed))); }
    });
  };
  trig.addEventListener("click", () => (list ? close() : open()));
  trig.addEventListener("keydown", (e) => { if (["ArrowDown", "ArrowUp", "Enter", " "].includes(e.key) && !list) { e.preventDefault(); open(); } });
  sync();
  return { el: trig, get: () => val, set };
}
