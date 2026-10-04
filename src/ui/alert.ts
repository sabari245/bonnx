import { cva, type VariantProps } from "class-variance-authority";
import { h, icon } from "../lib/dom";
import { cn } from "../lib/cn";
import { CircleAlert, Info } from "lucide";

const alertVariants = cva(
  "relative w-full rounded-lg border px-4 py-3 text-sm grid has-[>svg]:grid-cols-[calc(var(--spacing)*4)_1fr] grid-cols-[0_1fr] has-[>svg]:gap-x-3 gap-y-0.5 items-start [&>svg]:size-4 [&>svg]:translate-y-0.5 [&>svg]:text-current",
  {
    variants: { variant: { default: "bg-card text-card-foreground", destructive: "text-destructive bg-card [&>svg]:text-current *:data-[slot=alert-description]:text-destructive/90" } },
    defaultVariants: { variant: "default" },
  },
);
export function alert(title: string, description?: string | Node, p: VariantProps<typeof alertVariants> & { class?: string } = {}): HTMLElement {
  return h("div", { "data-slot": "alert", role: "alert", class: cn(alertVariants({ variant: p.variant }), p.class) },
    icon(p.variant === "destructive" ? CircleAlert : Info),
    h("div", { "data-slot": "alert-title", class: "col-start-2 line-clamp-1 min-h-4 font-medium tracking-tight" }, title),
    description && h("div", { "data-slot": "alert-description", class: "text-muted-foreground col-start-2 grid justify-items-start gap-1 text-sm [&_p]:leading-relaxed" }, description));
}
