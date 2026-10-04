"""Generate src/onnx/metadata.json (operator schemas) from the installed python `onnx` package.

    .venv/bin/python scripts/gen_onnx_metadata.py

Format (all keys short to keep the file small):
  { "v": <onnx version>, "types": [[type_str, ...], ...],            # shared type-constraint lists
    "ops": { "<domain>": { "<OpName>": [ version-record, ... ] } } } # ascending since_version
  version-record = { "v": since_version, "d"?: doc, "i"?: inputs, "o"?: outputs, "a"?: attributes, "t"?: constraints }
  A field is omitted when identical to the previous version's field (readers carry it forward).
    inputs/outputs   [[name, description, typeStr, option(0 single,1 optional,2 variadic)], ...]
    attributes       [[name, description, typeName, required(0/1), default?], ...]
    constraints      [[param, description, typesIndex], ...]
"""
import json
import sys
from pathlib import Path

import onnx
from onnx import defs, AttributeProto

OPT = {defs.OpSchema.FormalParameterOption.Single: 0, defs.OpSchema.FormalParameterOption.Optional: 1,
       defs.OpSchema.FormalParameterOption.Variadic: 2}


def default_value(a):
    v = a.default_value
    t = v.type
    T = AttributeProto
    if t == T.INT:
        return v.i
    if t == T.FLOAT:
        return round(v.f, 9)
    if t == T.STRING:
        return v.s.decode("utf-8", "replace")
    if t == T.INTS:
        return list(v.ints)
    if t == T.FLOATS:
        return [round(x, 9) for x in v.floats]
    if t == T.STRINGS:
        return [s.decode("utf-8", "replace") for s in v.strings]
    return None


def main():
    types, type_idx = [], {}

    def tix(lst):
        key = tuple(lst)
        if key not in type_idx:
            type_idx[key] = len(types)
            types.append(list(lst))
        return type_idx[key]

    ops = {}
    for s in sorted(defs.get_all_schemas_with_history(), key=lambda s: (s.domain, s.name, s.since_version)):
        rec = {
            "v": s.since_version,
            "d": s.doc or "",
            "i": [[p.name, p.description, p.type_str, OPT[p.option]] for p in s.inputs],
            "o": [[p.name, p.description, p.type_str, OPT[p.option]] for p in s.outputs],
            "a": [[a.name, a.description, AttributeProto.AttributeType.Name(a.type), int(a.required)]
                  + ([dv] if (dv := default_value(a)) is not None else [])
                  for a in sorted(s.attributes.values(), key=lambda a: a.name)],
            "t": [[c.type_param_str, c.description, tix(c.allowed_type_strs)] for c in s.type_constraints],
        }
        if s.deprecated:
            rec["x"] = 1
        lst = ops.setdefault(s.domain or "ai.onnx", {}).setdefault(s.name, [])
        prev = {}
        for r in lst:
            prev.update({k: r[k] for k in r if k in ("d", "i", "o", "a", "t")})
        out = {"v": rec["v"]}
        for k in ("d", "i", "o", "a", "t"):
            if prev.get(k) != rec[k]:
                out[k] = rec[k]
        if "x" in rec:
            out["x"] = 1
        lst.append(out)

    dest = Path(__file__).resolve().parent.parent / "src" / "onnx" / "metadata.json"
    data = {"v": onnx.__version__, "types": types, "ops": ops}
    dest.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    import gzip
    raw = dest.stat().st_size
    gz = len(gzip.compress(dest.read_bytes(), 9))
    print(f"wrote {dest} ({raw/1e6:.2f} MB, gzip {gz/1e3:.0f} KB); "
          f"{sum(len(v) for v in ops.values())} operators, {sum(len(r) for d in ops.values() for r in d.values())} versions",
          file=sys.stderr)


if __name__ == "__main__":
    main()
