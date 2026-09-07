#!/usr/bin/env python3
"""Recompute the @@ line counts of a unified diff, and verify them.

WHY THIS EXISTS
---------------
`integration/t5b-integration.patch` is a hand-maintained mirror of the hunks
that `apply_integration.py` applies by anchor. Editing one of its hunks changes
how many lines that hunk contains, and the `@@ -a,b +c,d @@` header carries
those counts. `git apply` checks them and refuses the whole patch when they
disagree -- so an edit that looks right and reads right produces a patch nobody
can apply, and the failure surfaces on a reviewer's machine rather than here.

This recomputes b and d from the hunk bodies, and c from the accumulated size
change of the hunks before it within the same file -- growing a hunk by thirteen
lines moves every later hunk's position in the NEW file by thirteen, and leaving
those stale is the second way this patch stops applying. `a`, the position in
the OLD file, is never touched: that describes the upstream tree this diff was
generated against, which is not present here, and computing it would be
guessing.

    --check   report disagreements and exit 1, changing nothing
    (default) rewrite the headers in place
"""
import re
import sys
from pathlib import Path

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def process(path: Path, check: bool) -> int:
    lines = path.read_text().splitlines(keepends=True)
    out, i, bad, fixed = [], 0, 0, 0
    delta = 0          # running (new - old) size change within the current file

    while i < len(lines):
        m = HUNK.match(lines[i].rstrip("\n"))
        if not m:
            # A new file resets the accumulated shift: hunk positions are
            # per-file, and carrying a delta across a `--- a/other.c` boundary
            # would corrupt every hunk after the first file.
            if lines[i].startswith("--- "):
                delta = 0
            out.append(lines[i]); i += 1; continue

        a, _b, _c, _d, tail = m.groups()
        body, j = [], i + 1
        while j < len(lines) and not HUNK.match(lines[j].rstrip("\n")) \
                and not lines[j].startswith(("diff ", "--- ", "+++ ", "Subject:")):
            body.append(lines[j]); j += 1

        # A context line counts on both sides, a '-' only on the old, a '+' only
        # on the new. Everything else -- "\ No newline at end of file", stray
        # blank lines at the tail of a hunk -- counts as neither.
        old = sum(1 for l in body if l.startswith((" ", "-")) or l.rstrip("\n") == "")
        new = sum(1 for l in body if l.startswith((" ", "+")) or l.rstrip("\n") == "")

        c = int(a) + delta
        delta += new - old
        want = f"@@ -{a},{old} +{c},{new} @@{tail}\n"
        if want != lines[i]:
            bad += 1
            if check:
                print(f"  {path.name}: hunk at -{a} says "
                      f"{lines[i].strip()!r}, content gives -{a},{old} +{c},{new}")
            else:
                fixed += 1
                lines[i] = want
        out.append(lines[i]); out.extend(body); i = j

    if check:
        if bad:
            print(f"\nFAIL -- {bad} hunk header(s) disagree with their content. "
                  f"`git apply` would reject this patch.")
            return 1
        print(f"  {path.name}: every hunk header matches its content.")
        return 0

    path.write_text("".join(out))
    print(f"  {path.name}: {fixed} header(s) rewritten, {len(lines)} lines.")
    return 0


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    check = "--check" in sys.argv
    if not args:
        # Both layouts, same file. See the note in check_patch_parity.py.
        root = Path(__file__).resolve().parent.parent
        for cand in ("integration/t5b-integration.patch",
                     "../bitnet-t5b/integration/t5b-integration.patch"):
            if (root / cand).exists():
                args = [str(root / cand)]
                break
        else:
            print("  t5b-integration.patch not found in either layout"
                  " -- skipping.")
            return 0
    rc = 0
    for a in args:
        p = Path(a)
        if not p.exists():
            print(f"  missing: {p}", file=sys.stderr); return 2
        rc |= process(p, check)
    return rc


if __name__ == "__main__":
    sys.exit(main())
