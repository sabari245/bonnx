import { icon } from "../lib/dom";
import { LoaderCircle } from "lucide";
import { cn } from "../lib/cn";
export const spinner = (cls?: string) => { const s = icon(LoaderCircle, cn("size-4 ui-spin", cls)); s.setAttribute("role", "status"); s.setAttribute("aria-label", "Loading"); return s; };
