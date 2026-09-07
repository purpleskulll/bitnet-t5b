#!/usr/bin/env python3
"""Do the integration script and the shipped patch describe the same change?

WHY THIS EXISTS
---------------
The llama.cpp integration exists twice, on purpose:

    scripts/apply_integration.py    applies the hunks by ANCHOR, tolerant of
                                    upstream line drift. The thing to run.
    integration/t5b-integration.patch   the same change as a unified diff, so a
                                    reviewer can read it without executing
                                    anything.

Two hand-maintained representations of one change is the exact shape that has
gone stale three times in this project already -- the paper's .tex behind its
.md, a packaged benchmark behind its source, the public tree's glue behind the
instrumentation a published result rests on. It went stale here too: adding the
symmetric SGEMM probe to the script left the patch file describing the previous
version of the dispatch site, and the divergence was found by reading, not by a
check.

So: every replacement body in the script must appear, line for line, among the
patch's added ('+') lines. That direction is the one that matters -- the script
is what runs, and a patch missing something the script does is a patch that
produces a different build.

It deliberately does NOT require the converse. The patch carries context lines,
file headers and hunk positions that have no counterpart in an anchored
replacement, and demanding symmetry there would mean encoding diff format into
this check for no gain.

Exit 0 when they agree, 1 when they do not.
"""
import re
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
PATCH  = locate("integration/t5b-integration.patch",
                "../bitnet-t5b/integration/t5b-integration.patch")

# Lines that are noise for this comparison: blank, or pure punctuation that
# appears in half the file anyway and would match by accident.
def significant(line: str) -> bool:
    s = line.strip()
    return len(s) > 12 and s not in {"*/", "/*", "{", "}", "});", "}));"}


def main() -> int:
    if PATCH is None or SCRIPT is None:
        print("  patch or integration script not found in either layout"
              " -- skipping.")
        return 0

    src = SCRIPT.read_text()
    # Added lines AND context lines. An anchored replacement has to carry the
    # surrounding lines it replaces, and a diff shows those as context (' '),
    # not as additions ('+'). Comparing against additions alone reported 27
    # perfectly correct context lines as missing -- the check was measuring the
    # difference between two file formats, not between two descriptions of a
    # change.
    added = set()
    for l in PATCH.read_text().splitlines():
        if l.startswith("+++") or l.startswith("---"):
            continue
        if l[:1] in ("+", " "):
            added.add(l[1:].strip())

    # The replacement bodies: element [2] of each (name, old, new) tuple in the
    # patch tables. Parsed from the AST, not matched with a regular expression.
    # The regex form was tried first and was wrong in the way that matters: it
    # swept in this script's own module docstring and every other triple-quoted
    # string in the file, reporting 650 "missing" lines that were prose. A
    # parser knows what a tuple is; a pattern only knows what quotes look like.
    #
    # Scope: the T5B_* tables only. The patch file's own header says it "covers
    # the SIX MODIFIED upstream files" of the t5b change; the script ALSO
    # carries the phase-3 sign application (SGEMM_PATCHES), the dead-row-skip
    # variant (T5S_*) and the per-layer FFN fix (BITNET_NFF_PATCHES), none of
    # which the patch claims to describe. Comparing everything reported 255
    # "missing" lines that were simply out of scope -- a check that cries wolf
    # gets switched off, so it is scoped to what the patch says it is.
    import ast
    WANT = ("T5B_GGML_H_PATCHES", "T5B_GGML_C_PATCHES", "T5B_CPU_PATCHES",
            "T5B_SGEMM_H_PATCHES", "T5B_SGEMM_CPP_PATCHES")
    bodies = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Assign):
            continue
        if getattr(node.targets[0], "id", None) not in WANT:
            continue
        for e in getattr(node.value, "elts", []):
            if isinstance(e, ast.Tuple) and len(e.elts) == 3:
                vals = [x.value if isinstance(x, ast.Constant) else None
                        for x in e.elts]
                if all(isinstance(v, str) for v in vals):
                    # What the hunk ADDS: lines in the replacement that are not
                    # already in the anchor. An anchored replacement quotes the
                    # lines it surrounds, and those are not part of the change
                    # -- requiring them to appear in the patch flagged five
                    # context lines whose only crime was that the diff was
                    # generated with a different neighbourhood. The change is
                    # the difference, and that is what is compared.
                    keep = set(l.strip() for l in vals[1].splitlines())
                    bodies.append("\n".join(
                        l for l in vals[2].splitlines() if l.strip() not in keep))

    missing, checked = [], 0
    for body in bodies:
        for line in body.splitlines():
            if not significant(line):
                continue
            checked += 1
            if line.strip() not in added:
                missing.append(line.strip())

    print("PATCH PARITY: apply_integration.py against t5b-integration.patch")
    print(f"  replacement lines checked   {checked}")
    print(f"  '+' lines in the patch      {len(added)}")

    if not missing:
        print("\nPASS -- every line the script inserts appears in the patch.")
        return 0

    print(f"\n  In the script but NOT among the patch's additions ({len(missing)}):")
    for m in missing[:20]:
        print(f"    {m}")
    if len(missing) > 20:
        print(f"    ... and {len(missing) - 20} more")
    print("\nFAIL -- the patch describes a different change from the script.")
    print("  The script is what runs. Update integration/t5b-integration.patch,")
    print("  then run tools/fix_patch_hunks.py so its @@ counts still match.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
