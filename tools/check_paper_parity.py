#!/usr/bin/env python3
"""Do the two renderings of the paper still say the same thing?

WHY THIS EXISTS
---------------
The paper is maintained as `paper/arxiv.md` and `paper/bitnet-baremetal-t5b.tex`.
The `.tex` is what gets submitted; the `.md` is what people read in the
repository. They are edited by hand, and they have drifted THREE times:

  1. The `.tex` fell an entire revision behind and a reviewer caught it.
  2. Fixed by copying, which fixed that revision and nothing structural.
  3. Caught again by the same reviewer, worse: `.tex` was missing the whole
     TQ1_0 comparison (the paper's most important new content), still carried
     the retracted claim that removing dead neurons "needs no format change and
     no kernel work" and is "unconditionally exact", and quoted 30% for a
     quantity measured at 43.6% two subsections earlier.

The right fix is one source. `pandoc` is not available here and installing it
was declined, so the fix available is this: make the drift MECHANICALLY VISIBLE
before it reaches a reviewer.

WHAT IT COMPARES, AND WHY NUMBERS
---------------------------------
Prose cannot be diffed across two markup languages without a converter. Numbers
can. Every drift above showed up as a number present in one file and absent from
the other -- 1.60755 and 1085565120 and 43.95 when §7.6 was missing, 0.751 and
2.35 when §8 was stale, 43.6 against 30 in the contradiction. So this extracts
every numeric literal from both files, normalises the two spellings to the same
token, and reports what appears in one and not the other.

It is a smoke detector and not a proof: prose can diverge with identical
numbers, and it says so rather than implying otherwise. What it catches is the
failure that actually happened, three times.

Numbers are normalised so the two markups compare equal:
    \\num{1187801280}, 1,187,801,280 and 1187801280   -> 1187801280
    \\SI{4.76}{\\percent}, 4.76%                       -> 4.76
    \\numrange{3.0}{4.0}                              -> 3.0 and 4.0

Exit 0 when they agree, 1 when they do not.
"""
import re
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "paper/arxiv.md"
TEX = ROOT / "paper/bitnet-baremetal-t5b.tex"

# Numbers that legitimately appear in one file only, each with the reason. An
# entry here is a claim that this difference is intended; adding one without a
# reason defeats the check.
ALLOWED = {
    # LaTeX plumbing: font sizes, column widths, version numbers in the preamble.
    "0.36", "0.56", "0.44", "0.52", "0.62", "1.0", "0.5", "11", "12", "10",
    # Section numbers used to be listed here one by one. They are handled
    # structurally now (see STRUCTURAL below): the Markdown is generated and its
    # section numbers come from LaTeX's own counters, so enumerating them was
    # both redundant and a list that silently went stale when the numbering
    # moved.
}

NUM = re.compile(r"\d+(?:[.,]\d+)*")

# Four classes of digit that are pointers or plumbing rather than claims, and
# that legitimately appear in one rendering only. Stripping them beats listing
# their values in ALLOWED, which would have to grow every time a date or an
# equation is added.
STRUCTURAL = [
    (re.compile(r"\d{4}-\d{2}-\d{2}"), " "),        # ISO dates in the front matter
    (re.compile(r"\\tag\{\d+\}"), " "),             # manual equation tags; LaTeX numbers its own
    (re.compile(r"\.(?:txt|md|c|h|py|sh|cpp|json):\d+"), " "),  # file:line pointers
    (re.compile(r"^#{1,6}\s+[\d.]+[a-z]?\s", re.M), " "),       # markdown heading numerals
    # Section cross-references. The Markdown is GENERATED from the LaTeX by
    # tools/tex_to_md.py, which resolves \ref{sec:x} to the number LaTeX
    # assigns; the .tex therefore never contains the digits. Listing each one in
    # ALLOWED worked until the numbering shifted -- which it did the moment the
    # generator replaced the hand-written file and revealed that every "§7.6"
    # in it had pointed at what LaTeX numbers 7.7.
    (re.compile(r"§\s?\d+(?:\.\d+)?"), " "),
]


def numbers(text: str, is_tex: bool) -> Counter:
    # Equation cross-references. Markdown has to write the number -- "by (18)"
    # -- where LaTeX writes \eqref and lets the compiler assign it, so these are
    # in one rendering only by construction.
    #
    # A blanket strip of "(N)" would be wrong and this was checked rather than
    # assumed: the markdown also contains "the type identifier used (43)", a
    # real claim about a GGML type number. So only those N that ACTUALLY LABEL
    # an equation in this file are removed -- a reference must refer to
    # something. 43 tags no equation and survives; 18 does and does not.
    tagged = set(re.findall(r"\\tag\{(\d+)\}", text))
    if tagged:
        text = re.sub(r"\((\d+)\)",
                      lambda m: " " if m.group(1) in tagged else m.group(0), text)
        text = re.sub(r"\\tag\{\d+\}", " ", text)

    for pat, rep in STRUCTURAL:
        text = pat.sub(rep, text)
    if is_tex:
        # Unwrap the siunitx forms first so their payloads survive as plain
        # numbers; \numrange and \SIrange carry two.
        text = re.sub(r"\\num(?:range)?\{([^}]*)\}\{([^}]*)\}", r" \1 \2 ", text)
        text = re.sub(r"\\SIrange\{([^}]*)\}\{([^}]*)\}\{[^}]*\}", r" \1 \2 ", text)
        text = re.sub(r"\\(?:num|SI)\{([^}]*)\}(?:\{[^}]*\})?", r" \1 ", text)
        # Strip the preamble: package options and geometry are not claims.
        body = text.split(r"\begin{document}", 1)
        text = body[1] if len(body) > 1 else text
        # Bibliography plumbing: the {99} width argument and every citation key.
        # \bibitem{bitnet158} is a label, not the number 158, and reading it as
        # one made this check report a difference that did not exist.
        text = re.sub(r"\\begin\{thebibliography\}\{[^}]*\}", " ", text)
        text = re.sub(r"\\(?:bibitem|cite|label|ref|eqref)\{[^}]*\}", " ", text)
        # LaTeX control sequences can contain digits that are not numbers.
        text = re.sub(r"\\[a-zA-Z]+", " ", text)
    else:
        # The markdown carries a few \SI{} forms verbatim; treat them the same.
        text = re.sub(r"\\(?:num|SI)\{([^}]*)\}(?:\{[^}]*\})?", r" \1 ", text)
        # Fenced code blocks are commands, not claims.
        text = re.sub(r"```.*?```", " ", text, flags=re.S)
        # Markdown table separators (|---:|) and heading hashes.
        text = re.sub(r"^\s*\|[\s:|-]+\|\s*$", " ", text, flags=re.M)

    # LaTeX digit separators, which appear in the markdown too because its maths
    # is LaTeX: 32{,}512 and 32\,512 are both the number 32512. Removing these
    # before matching stops the extractor from inventing tokens like "024" out
    # of one number written with separators.
    text = text.replace("{,}", "").replace("\\,", "").replace("{\\,}", "")

    out = Counter()
    for m in NUM.finditer(text):
        tok = m.group(0).replace(",", "")
        # Drop a trailing decimal zero difference: 2.0 and 2.00 are the same
        # claim written twice, and chasing that is noise.
        if "." in tok:
            tok = tok.rstrip("0").rstrip(".") or "0"
        if tok in ALLOWED or len(tok) == 0:
            continue
        # Single digits are almost all structural (list markers, small counts in
        # prose) and produce noise without catching drift.
        if len(tok) < 2:
            continue
        out[tok] += 1
    return out


# The paper states how many corrections there have been and points at
# CHANGELOG.md for the list. That is one number living in two files, maintained
# by hand, and it drifted the first time anyone looked: the paper said "Eight"
# while the CHANGELOG had reached eleven, because items 9, 10 and 11 were
# appended without anyone touching a sentence three hundred lines away in
# another file. A count that appears twice needs a check or it drifts again.
# Spelled out to thirty, and a digit form accepted too. The first version of
# this stopped at fifteen and fired on "Seventeen" with the message "the
# correction count has drifted" -- which was false: the counts agreed and the
# PARSER had run out. A check whose failure message names the wrong cause is
# worse than one that stays silent, so the two are now distinguished below.
WORDS = {w: i + 1 for i, w in enumerate(
    "one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one "
    "twenty-two twenty-three twenty-four twenty-five twenty-six twenty-seven "
    "twenty-eight twenty-nine thirty".split())}


def correction_count() -> int:
    """How many numbered corrections CHANGELOG.md actually carries, or -1."""
    ch = ROOT / "CHANGELOG.md"
    if not ch.exists():
        return -1
    return len(re.findall(r"^\*\*(\d+)\.", ch.read_text(), re.M))


def check_correction_count() -> list:
    """The count each rendering states, against the count the CHANGELOG holds."""
    n = correction_count()
    if n < 0:
        return []
    bad = []
    for doc in (TEX, MD):
        if not doc.exists():
            continue
        m = re.search(r"\b([A-Za-z]+) claims in earlier drafts", doc.read_text())
        if not m:
            bad.append(f"{doc.name}: no 'N claims in earlier drafts' sentence")
            continue
        word = m.group(1)
        said = WORDS.get(word.lower())
        if said is None and word.isdigit():
            said = int(word)
        if said is None:
            bad.append(f"{doc.name}: PARSER LIMIT, not a drift -- cannot read "
                       f"'{word}' as a number. Extend WORDS; the counts may "
                       f"well agree.")
        elif said != n:
            bad.append(f"{doc.name}: says {m.group(1)} ({said}), "
                       f"CHANGELOG.md has {n}")
    return bad


def main() -> int:
    for p in (MD, TEX):
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 2

    md = numbers(MD.read_text(), is_tex=False)
    tex = numbers(TEX.read_text(), is_tex=True)

    only_md = sorted(set(md) - set(tex))
    only_tex = sorted(set(tex) - set(md))

    drift = check_correction_count()

    print("PAPER PARITY: arxiv.md against bitnet-baremetal-t5b.tex")
    print(f"  corrections: CHANGELOG.md has {correction_count()}, "
          f"both renderings agree" if not drift else
          f"  corrections: CHANGELOG.md has {correction_count()}, THE PAPER DOES NOT AGREE")
    print(f"  distinct numeric claims   md {len(md):4d}   tex {len(tex):4d}")
    print(f"  shared                    {len(set(md) & set(tex)):4d}")

    if drift:
        print(f"\n  The correction count has drifted ({len(drift)}):")
        for b in drift:
            print(f"    {b}")

    if not only_md and not only_tex and not drift:
        print("\nPASS -- every numeric claim appears in both renderings.")
        print("  This does NOT prove the prose agrees. It proves the failure")
        print("  that has happened three times has not happened again.")
        return 0

    if only_md:
        print(f"\n  In arxiv.md but NOT in the .tex ({len(only_md)}):")
        for t in only_md:
            print(f"    {t}")
    if only_tex:
        print(f"\n  In the .tex but NOT in arxiv.md ({len(only_tex)}):")
        for t in only_tex:
            print(f"    {t}")

    if drift and not (only_md or only_tex):
        print("\nFAIL -- the paper states a correction count that CHANGELOG.md")
        print("  does not hold. One of the two was edited without the other.")
        return 1

    print("\nFAIL -- the two renderings carry different numbers.")
    print("  The .tex is what gets submitted. If a claim is only in the .md,")
    print("  it is not in the paper. Port it, or add the token to ALLOWED with")
    print("  the reason it legitimately differs.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
