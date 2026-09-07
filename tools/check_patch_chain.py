#!/usr/bin/env python3
"""Do the later patch tables still anchor on what the earlier ones emit?

WHY THIS EXISTS
---------------
`apply_integration.py` patches by ANCHOR: each entry is (name, old, new), and
`old` must be found verbatim in the file before `new` replaces it. The tables
are applied in order, and they CHAIN -- T5S patches a tree that T5B has already
patched, so several T5S anchors quote text that a T5B replacement produced.

That coupling is invisible until someone runs the script, which needs a
llama.cpp checkout and a container. Adding the symmetric SGEMM probe to
T5B_CPU_PATCHES changed `if (!(sg_is_t5b` into
`const bool sg_ok = (sg_is_t5b`, and every T5S anchor quoting the old form
stopped matching. The script would have failed on the next integration run, on
someone else's machine, with a message about an anchor not being found and no
indication that a change three hundred lines away had caused it.

WHAT IT CHECKS
--------------
For every anchor in a downstream table that mentions an identifier introduced by
an upstream table (t5b, sg_is_t5b, sg_ok, llamafile_sgemm_t5b, ...), that anchor
must appear verbatim in some upstream table's replacement text. An anchor that
does not is either broken or is quoting the pristine upstream file, and this
prints which so a reader can tell the two apart.

It cannot verify anchors against the real upstream tree, which is not present
here -- that is what `apply_integration.py` itself does when run. This checks
the part that is checkable without it, which is the part that breaks silently.

Exit 0 when the chain holds, 1 when it does not.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# This file runs UNCHANGED in both trees, which is deliberate: it is copied
# between them by scripts/sync_public.sh, and a copy that needed editing on
# arrival is a copy that drifts. The two layouts differ --
#
#   private:  scripts/apply_integration.py   ../bitnet-t5b/integration/*.patch
#   public:   integration/apply_integration.py   integration/*.patch
#
# -- so paths are LOCATED rather than assumed, and a miss says which candidates
# were tried instead of failing on a stack trace.
def locate(*candidates):
    """First existing path among the candidates, or None."""
    for c in candidates:
        p = (ROOT / c).resolve()
        if p.exists():
            return p
    return None

SCRIPT = locate("scripts/apply_integration.py", "integration/apply_integration.py")

# Applied in this order by apply_phase3(); a table may only anchor on the output
# of tables listed before it.
ORDER = [
    "SGEMM_PATCHES",
    "T5B_GGML_H_PATCHES", "T5B_GGML_C_PATCHES", "T5B_CPU_PATCHES",
    "T5B_SGEMM_H_PATCHES", "T5B_SGEMM_CPP_PATCHES",
    "T5S_GGML_H_PATCHES", "T5S_GGML_C_PATCHES", "T5S_CPU_PATCHES",
    "T5S_SGEMM_H_PATCHES", "T5S_SGEMM_CPP_PATCHES",
    "BITNET_NFF_PATCHES",
]

# An anchor containing one of these was produced by an earlier table rather than
# by upstream, so it must be traceable to one.
INTRODUCED = ("sg_is_t5b", "sg_ok", "sg_probe_t0", "llamafile_sgemm_t5b",
              "ggml_t5b_row_bytes", "GGML_TYPE_T5B", "bitnet_t5b",
              "sg_t5b_rowb", "bitnet_sgemm_cycles_add")


def tables(src: str) -> dict:
    out = {}
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Assign):
            continue
        name = getattr(node.targets[0], "id", None)
        if not name or not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        ent = []
        for e in node.value.elts:
            if isinstance(e, ast.Tuple) and len(e.elts) == 3:
                v = [x.value if isinstance(x, ast.Constant) else None for x in e.elts]
                if all(isinstance(z, str) for z in v):
                    ent.append(v)
        if ent:
            out[name] = ent
    return out


def main() -> int:
    if SCRIPT is None:
        print("  apply_integration.py not found in either layout -- skipping.")
        return 0
    tab = tables(SCRIPT.read_text())
    seen_new: list[str] = []
    broken, traced, upstream = [], 0, 0

    print("PATCH CHAIN: later anchors against earlier replacements")
    for tname in ORDER:
        for name, old, new in tab.get(tname, []):
            if any(tok in old for tok in INTRODUCED):
                if any(old in prev for prev in seen_new):
                    traced += 1
                else:
                    broken.append((tname, name, old.strip().splitlines()[0][:72]))
            else:
                upstream += 1
        for _n, _o, new in tab.get(tname, []):
            seen_new.append(new)

    print(f"  anchors quoting an earlier replacement   {traced}")
    print(f"  anchors quoting pristine upstream        {upstream}")

    if not broken:
        print("\nPASS -- every chained anchor is produced by a table before it.")
        return 0

    print(f"\n  Anchors that no earlier table emits ({len(broken)}):")
    for t, n, head in broken:
        print(f"    {t} :: {n}")
        print(f"        {head}")
    print("\nFAIL -- these will not be found when the script runs. An earlier")
    print("  table's replacement text changed and these still quote the old form.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
