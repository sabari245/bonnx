# UI kit

shadcn/ui (new-york, neutral base, blue primary) re-implemented as **vanilla TypeScript** on Tailwind v4.
Class recipes and CSS tokens are shadcn's; behaviour (ARIA, keyboard, focus trap, Escape, click-outside) is hand-written.
Import everything from `src/ui` (`import { button, dialog, toast } from "@/ui"` or relative). Styles: `import "./styles/globals.css"` once.
Gallery (visual check, light/dark): `gallery.html` (`?theme=dark`, `?open=dialog|sheet|palette|dropdown|select|popover|ctx|toast|tooltip`).

Conventions: components are functions returning an `HTMLElement`, or a small controller `{ el, get, set, … }` when they hold state.
Optional `class` props go through `cn()` (tailwind-merge) so overrides win. Build DOM with `h(tag, props, ...children)` from `lib/dom`.
Theme: `initTheme()` once at startup; `setTheme("light"|"dark"|"system")`, `getTheme()`, `resolvedTheme()`, `cycleTheme()`; a `themechange` event fires on `window`.

## Primitives
| fn | notes |
|---|---|
| `button(label, {variant,size,icon,onClick,pressed,title,ariaLabel,disabled,class})` | `buttonVariants` (cva) exported; variants default/destructive/outline/secondary/ghost/link; sizes default/sm/lg/icon/icon-sm |
| `badge(label,{variant,class,style})`, `kbd(text)`, `kbdGroup("Ctrl+K")`, `separator(orientation)`, `label(text,{for})` | |
| `input({value,placeholder,type,onInput,onChange})`, `textarea(...)` | `inputClass` exported |
| `card(cls,...)`, `cardHeader/Title/Description/Action/Content/Footer` | |
| `alert(title, description, {variant})`, `skeleton(cls)`, `spinner(cls)`, `progress(value\|null)` → `{el,set}` | |
| `emptyState({icon,title,description,action})`, `breadcrumb([{label,onClick}])` | |
| `table(cls,...)`, `tableHeader/Body/Row/Head/Cell`, `scrollArea(cls,...children)` | |

## Controls (return controllers)
`switchControl({checked,onChange})`, `checkbox({checked: boolean|"indeterminate"})`, `slider({min,max,step,value,onInput,onCommit})`,
`toggle(label,{pressed,icon,variant})`, `toggleGroup(items,{type:"single"|"multiple",value,onChange})`,
`select(options,{value,onChange})` (supports groups), `tabs(items,{value,onValueChange})` (lazy content via function),
`collapsible({title,open},...content)`, `accordion(items,{type})`.

## Overlays
- `tooltip(el, text|()=>text, {side,delay})` → detach fn
- `popover(trigger, content|()=>node, {side,align,class})` → `{open,close,toggle,isOpen}`
- `dropdownMenu(trigger, items|()=>items)`, `contextMenu(target, items|(e)=>items|null)` → `{openAt(x,y,items)}`, `showMenu(items, anchor|{x,y})`.
  Item model: `{label,icon,shortcut,onSelect,disabled,destructive}`, `{type:"checkbox",label,checked,onChange}`, `{type:"radio",group,value,label,checked,onSelect}`, `{type:"separator"}`, `{type:"label",label}`, `{type:"sub",label,items}`.
- `dialog({title,description,content,footer,size,onClose})` → `{open,close,body,setTitle}`; `confirmDialog(title,desc,label)` → Promise<boolean>
- `sheet({title,width,modal,mount,storageKey,onResize})` → `{open,close,setContent,setTitle,body,getWidth,setWidth}` — resizable right panel; non-modal by default (docked inspector).
- `command({items,filter,onQuery,onSelect,onActive,maxItems})` → `{el,input,setItems,setQuery}`: inline fuzzy list; `filter:false` lets the caller supply pre-filtered items (large graphs). `commandDialog(opts)` → `{open,close,command}` for Ctrl/Cmd-K. `fuzzy(q,text)` exported. Item: `{id,label,hint,group,icon,swatch,keywords,shortcut,onSelect}`.
- `toast({title,description,variant:"default"|"success"|"destructive",duration,action})` → dismiss fn

## Layout / data
- `createVirtualList({rowHeight,count,render(i)→HTMLElement,overscan})` → `{el,setCount,refresh,scrollToIndex,getRange}`; millions of rows (scroll range is remapped above 16M px).
- `createVirtualTable({columns:[{header,width,align}],count,cell(row,col),rowHeight,onRowClick})` — sticky header, horizontal scroll, same controller.
- `resizablePanels({a,b,size,direction,storageKey})`, `dragResize(handle,{axis,get,set,min,max})`.

## Graph tokens (read with getComputedStyle)
`--graph-{bg,grid,edge,edge-hi,node-bg,node-row,node-border,node-text,node-subtext,grey,grey-text,sel}`,
`--cat-{layer,activation,norm,pool,math,reduce,shape,quant,control,other,input,const}` (white header text is readable in both themes),
`--ramp-0..4` (light theme: light→dark; dark theme: dim→bright so large values stand out).
