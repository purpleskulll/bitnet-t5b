#!/usr/bin/env python3
"""Does the four-page version still agree with the paper it was cut from?

WHY THIS EXISTS
---------------
There are now two papers. `paper/bitnet-baremetal-t5b.tex` is the long one,
which spends more lines conceding prior art than claiming novelty and is the
right document for the repository. `paper/short/t5b-short.tex` is the
submission.

Neither length is written down here, deliberately. The first version of this
docstring said "27 pages"; it was corrected to 28, and within the same day a
wave of edits made it 29. A page count is a number maintained by hand in three
files -- this docstring, its printed header, and a comment at the top of
t5b-short.tex -- which is precisely the shape of drift the rest of this file
exists to catch. So `page_counts()` reads both from the "Output written on ...
(N pages" line the builds actually emit, and the program prints what it read.
A derived number cannot go stale.

Two documents drift. This project has been bitten by that four times in one
week: a Markdown rendering a full revision behind its LaTeX, a public tree whose
evidence files were six of eight stale, a benchmark message naming targets that
had moved, and an evidence file silently overwritten by a different run. Each
was found by a person, none by a check, and each cost a day.

So the short paper is gated the same way the Markdown is. Every numeric claim it
makes must also appear in the long one. Correct a figure in the long paper and
forget the short one and this fails; write a number into the short one that the
long one does not support and this fails too.

WHAT IT DOES NOT DO, stated because a check that is believed to do more than it
does is worse than no check. It does not compare prose, it cannot tell whether a
number means the same thing in both places, and it says nothing about what the
short paper leaves out -- cutting is the point of it. The direction is one-way
on purpose: the long paper may carry figures the short one does not.

The extractor is imported from check_paper_parity rather than copied. A second
copy of that regex is a second thing to drift, which is the failure this file
exists to prevent.

Exit 0 when every figure in the short paper is in the long one, 1 when not.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

LONG = ROOT / "paper/bitnet-baremetal-t5b.tex"
SHORT = ROOT / "paper/short/t5b-short.tex"

# Figures the short paper is allowed to carry alone, each with the reason.
# An entry here is a claim that the figure is legitimately short-only, not a way
# to silence the check.
ALLOWED = {
    # none yet; the short paper was written from the long one's numbers.
}


def page_counts() -> tuple:
    """(long, short) page counts, read from the builds rather than asserted.

    LaTeX writes "Output written on X.pdf (N pages, M bytes)." on every
    successful run. Reading N there is the only spelling of the page count that
    cannot drift: it IS the build. Returns None for a document whose .log is
    absent or which did not finish -- a missing count is reported as missing
    and never guessed at.
    """
    import re
    out = []
    for tex in (LONG, SHORT):
        log = tex.with_suffix(".log")
        n = None
        if log.exists():
            m = re.search(r"Output written on \S+ \((\d+) pages",
                          log.read_text(errors="replace"))
            if m:
                n = int(m.group(1))
        out.append(n)
    return tuple(out)


def main() -> int:
    if not SHORT.exists():
        # NOT a skip, deliberately. A cold clone of the public repository did
        # not carry paper/short/ at all, so this branch fired on every run
        # there: the gate exited 0 having compared nothing and bought a green
        # tick with no coverage. That is the precise failure this project has a
        # standing rule about -- a check that cannot fail is worse than no
        # check, because it is believed. The short paper and this gate must
        # ship together or not at all.
        print(f"  {SHORT.relative_to(ROOT)} is MISSING, so nothing was checked.",
              file=sys.stderr)
        print("  This is a failure and not a skip: the gate is vacuous without",
              file=sys.stderr)
        print("  the paper. Ship the short paper alongside it, or retire both.",
              file=sys.stderr)
        return 1
    if not LONG.exists():
        print("  the long paper is missing; nothing to check against.", file=sys.stderr)
        return 1

    try:
        from check_paper_parity import numbers
    except ImportError as e:
        print(f"  cannot import the shared extractor: {e}", file=sys.stderr)
        return 1

    short = numbers(SHORT.read_text(), is_tex=True)
    long_ = numbers(LONG.read_text(), is_tex=True)

    missing = sorted(t for t in short if t not in long_ and t not in ALLOWED)

    # The counts come from the build logs, which are not shipped -- a reader who
    # cloned the public tree gets the line without them rather than a guess.
    nlong, nshort = page_counts()
    where = (f" ({nshort} pages against {nlong})" if nshort and nlong else
             "  [page counts unavailable: the build logs are not in this tree]")
    print(f"SHORT PARITY: the short version against the paper it was cut "
          f"from{where}")
    print(f"  figures in the short paper   {len(short):4d}")
    print(f"  of those, also in the long   {len(short) - len(missing):4d}")
    print(f"  allowed to differ            {len(ALLOWED):4d}")

    if not missing:
        print("\nPASS -- every figure the short paper quotes is in the long one.")
        print("  This does NOT check that they mean the same thing there, and it")
        print("  says nothing about what the short paper leaves out.")
        return 0

    print(f"\n  In the short paper but NOT in the long one ({len(missing)}):")
    for t in missing:
        print(f"    {t}")
    print("\nFAIL -- the submission carries a figure its own evidence document")
    print("  does not. Either the long paper was corrected and this was not, or")
    print("  the short one states something nothing backs.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
