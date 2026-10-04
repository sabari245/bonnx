export type Theme = "light" | "dark" | "system";
const KEY = "bonnx-theme";
let current: Theme = "system";

function read(): Theme {
  try {
    const v = localStorage.getItem(KEY);
    if (v === "light" || v === "dark" || v === "system") return v;
  } catch {
    /* storage blocked */
  }
  return "system";
}

const mq = typeof matchMedia === "function" ? matchMedia("(prefers-color-scheme: dark)") : null;

export function resolvedTheme(t: Theme = current): "light" | "dark" {
  return t === "system" ? (mq?.matches ? "dark" : "light") : t;
}

function apply(notify: boolean): void {
  const dark = resolvedTheme() === "dark";
  const root = document.documentElement;
  const changed = root.classList.contains("dark") !== dark;
  root.classList.toggle("dark", dark);
  if (notify || changed) window.dispatchEvent(new CustomEvent("themechange", { detail: { theme: current, resolved: resolvedTheme() } }));
}

export function getTheme(): Theme {
  return current;
}

export function setTheme(t: Theme): void {
  current = t;
  try {
    localStorage.setItem(KEY, t);
  } catch {
    /* ignore */
  }
  apply(true);
}

/** Call once at startup (before first paint ideally). Follows OS changes while theme === "system". */
export function initTheme(): void {
  current = read();
  apply(false);
  mq?.addEventListener("change", () => current === "system" && apply(true));
}

/** next in light → dark → system cycle, handy for a toggle button */
export function cycleTheme(): Theme {
  const next: Theme = current === "light" ? "dark" : current === "dark" ? "system" : "light";
  setTheme(next);
  return next;
}
