import { h } from "../lib/dom";
import { cn } from "../lib/cn";

type C = (string | Node | null | undefined | false)[];
export const card = (cls?: string, ...c: C) => h("div", { "data-slot": "card", class: cn("bg-card text-card-foreground flex flex-col gap-6 rounded-xl border py-6 shadow-sm", cls) }, ...c);
export const cardHeader = (cls?: string, ...c: C) => h("div", { "data-slot": "card-header", class: cn("@container/card-header grid auto-rows-min grid-rows-[auto_auto] items-start gap-1.5 px-6 has-data-[slot=card-action]:grid-cols-[1fr_auto] [.border-b]:pb-6", cls) }, ...c);
export const cardTitle = (text: string | Node, cls?: string) => h("div", { "data-slot": "card-title", class: cn("leading-none font-semibold", cls) }, text);
export const cardDescription = (text: string | Node, cls?: string) => h("div", { "data-slot": "card-description", class: cn("text-muted-foreground text-sm", cls) }, text);
export const cardAction = (cls?: string, ...c: C) => h("div", { "data-slot": "card-action", class: cn("col-start-2 row-span-2 row-start-1 self-start justify-self-end", cls) }, ...c);
export const cardContent = (cls?: string, ...c: C) => h("div", { "data-slot": "card-content", class: cn("px-6", cls) }, ...c);
export const cardFooter = (cls?: string, ...c: C) => h("div", { "data-slot": "card-footer", class: cn("flex items-center px-6 [.border-t]:pt-6", cls) }, ...c);
