import { cva, type VariantProps } from "class-variance-authority";
import { h } from "../lib/dom";
import { cn } from "../lib/cn";

export const badgeVariants = cva(
  "inline-flex items-center justify-center rounded-md border px-2 py-0.5 text-xs font-medium w-fit whitespace-nowrap shrink-0 [&>svg]:size-3 gap-1 [&>svg]:pointer-events-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] transition-[color,box-shadow] overflow-hidden",
  {
    variants: {
      variant: {
        default: "border-transparent bg-primary text-primary-foreground",
        secondary: "border-transparent bg-secondary text-secondary-foreground",
        destructive: "border-transparent bg-destructive text-white dark:bg-destructive/60",
        outline: "text-foreground",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export function badge(label: string | Node, p: VariantProps<typeof badgeVariants> & { class?: string; style?: string } = {}): HTMLSpanElement {
  return h("span", { "data-slot": "badge", class: cn(badgeVariants({ variant: p.variant }), p.class), style: p.style }, label);
}
