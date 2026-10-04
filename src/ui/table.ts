import { h } from "../lib/dom";
import { cn } from "../lib/cn";

type C = (string | Node | null | undefined | false)[];
/** Returns the scroll container; the <table> is `.firstElementChild`. */
export const table = (cls?: string, ...c: C) =>
  h("div", { "data-slot": "table-container", class: "ui-scroll relative w-full overflow-x-auto" }, h("table", { "data-slot": "table", class: cn("w-full caption-bottom text-sm", cls) }, ...c));
export const tableHeader = (...c: C) => h("thead", { "data-slot": "table-header", class: "[&_tr]:border-b" }, ...c);
export const tableBody = (...c: C) => h("tbody", { "data-slot": "table-body", class: "[&_tr:last-child]:border-0" }, ...c);
export const tableRow = (cls?: string, ...c: C) => h("tr", { "data-slot": "table-row", class: cn("hover:bg-muted/50 data-[state=selected]:bg-muted border-b transition-colors", cls) }, ...c);
export const tableHead = (text: string | Node, cls?: string) => h("th", { "data-slot": "table-head", class: cn("text-foreground h-10 px-2 text-left align-middle font-medium whitespace-nowrap", cls) }, text);
export const tableCell = (text: string | Node | null, cls?: string) => h("td", { "data-slot": "table-cell", class: cn("p-2 align-middle whitespace-nowrap", cls) }, text);
