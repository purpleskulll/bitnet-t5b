#!/usr/bin/env python3
"""Do the decode's stated arithmetic properties actually hold over all 256 bytes?

WHY THIS EXISTS
---------------
Every other check in this repository tests one artefact against another: the
`.md` against the `.tex`, a path against the tree a reader clones, a patch hunk
against the anchor an earlier hunk emits. That family cannot see a claim that
nothing else in the repository disagrees with.

CHANGELOG.md item 9 is one. `ternary_t5b.c` asserted, in a comment marked
"verified exhaustively", that every digit lies in {0,1,2} for every one of the
256 possible byte values. It does not: 243..255 all decode to d4 = 3, because
floor(x/81) = 3 there. The file's own header said so correctly eleven lines
away, the kernel was and is correct, and no gate fired -- because the wrong
thing was the justification, and a justification has no artefact to disagree
with. It mattered because the digit range is exactly what the int16 accumulator
bound rests on: 2*128*2 = 512 per plane, 5*512 = 2560 per block, twelve blocks.

So this check re-derives the arithmetic instead of comparing text. It reads the
four magic constants and the fold factor OUT of the source rather than
hard-coding them -- a check carrying its own copy of the number it is checking
proves only that someone typed it twice -- and then evaluates every property the
source and the paper claim, over the whole 256-byte domain, by exhaustion.

WHAT IT CHECKS, all exhaustive, none sampled
--------------------------------------------
1. MAGIC EXACTNESS.  mulhi(x, M_j) == floor(x / 3^j) for every byte, for all
   four constants. This is the property the paper's magic-bound table asserts;
   here it is evaluated rather than tabulated.
2. BASE-3 CORRECTNESS.  On the legal domain x <= 242 the five digits are the
   true base-3 digits of x. This is what makes the format lossless.
3. BORROW FREEDOM.  d_k = x_k - 3*x_{k+1} never goes negative, for any of the
   256 values including the foreign ones. This is the property `vpsubb` needs,
   it is weaker than the one the comment claimed, and unlike that one it holds.
4. THE DIGIT RANGE, AND ITS EXACT EXCEPTION SET.  Digits are in {0,1,2} on
   x <= 242 and the set of bytes that leave that range is EXACTLY {243..255},
   each by d4 = 3. Asserting the exception set rather than its size means a
   future change to a magic constant that moves the boundary fails here.
5. THE ACCUMULATOR BOUND.  From the per-plane maxima it recomputes the legal
   bound (2560), the foreign-byte bound (2816), and checks that the fold factor
   in the header is the largest one that fits int16 -- and that one more does
   not. This is equation (fold) of the paper, evaluated.

NEGATIVE CONTROLS
-----------------
`--self-test` re-runs everything against three deliberately broken decoders,
one per property class, and every one of them must be caught. A control that
passes is not a control, so a survivor fails this check rather than being
reported as a curiosity.

  m3 2428 -> 2427   breaks magic exactness, and through it base-3 correctness
  m4  811 ->  813   exact for every byte except one: it first fails at x = 242,
                    the very last legal value, so it is the sharpest available
                    test of whether the domain is really being swept
  fold  12 -> 13    breaks nothing arithmetic and overflows int16, which is the
                    one failure here that is about the kernel rather than the
                    decode

The first control this file shipped with was 811 -> 810, on the reasoning that
the mutation suite had proved it equivalent only on x <= 242. That reasoning was
wrong and the control did not fire: sweeping the constants shows 810, 811 and
812 are ALL exact for /81 over the whole 256-byte domain, so 810 is equivalent
everywhere and not merely on the legal part. That is the explanation for one of
the three equivalent mutants `mutation_test.sh` reports. It is also why the
controls above were chosen by sweeping rather than by guessing: each shipped
constant is the SMALLEST exact multiplier for its divisor, so subtracting one
breaks three of the four -- and m4 is the exception, which is precisely the one
a guess would have picked.

Exit 0 when every property holds, 1 when one does not.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Both layouts, located rather than assumed -- this file is copied to the public
# tree by scripts/sync_public.sh and a copy that needs editing on arrival drifts.
def locate(*candidates):
    for c in candidates:
        p = (ROOT / c).resolve()
        if p.exists():
            return p
    return None

SRC = locate("src_modifications/ternary_t5b.c", "src/ternary_t5b.c")
HDR = locate("src_modifications/ternary_t5b.h", "src/ternary_t5b.h")

DIGITS = 5
A_MAX = 128          # |a| <= 128, the full int8 range, as the header derives it
INT16_MAX = 32767


def read_constants():
    """The four magic multipliers and the fold factor, taken from the source."""
    src = SRC.read_text()
    magics = {}
    for j, want in ((1, r"m1\s*=\s*_mm256_set1_epi16\((\d+)\)"),
                    (2, r"m2\s*=\s*_mm256_set1_epi16\((\d+)\)"),
                    (3, r"m3\s*=\s*_mm256_set1_epi16\((\d+)\)"),
                    (4, r"m4\s*=\s*_mm256_set1_epi16\((\d+)\)")):
        m = re.search(want, src)
        if not m:
            print(f"  could not read magic m{j} from {SRC}", file=sys.stderr)
            sys.exit(2)
        magics[j] = int(m.group(1))
    m = re.search(r"#define\s+TERNARY_T5B_FOLD\s+(\d+)", HDR.read_text())
    if not m:
        print(f"  could not read TERNARY_T5B_FOLD from {HDR}", file=sys.stderr)
        sys.exit(2)
    return magics, int(m.group(1))


def decode(x, magics):
    """The kernel's five digit planes for one byte, in the kernel's own order."""
    mulhi = lambda v, m: (v * m) >> 16
    q = {j: mulhi(x, magics[j]) for j in (1, 2, 3, 4)}
    return [x - 3 * q[1], q[1] - 3 * q[2], q[2] - 3 * q[3], q[3] - 3 * q[4], q[4]], q


def evaluate(magics, fold):
    """Every property, over the whole domain. Returns a list of failures."""
    bad = []
    plane_max = [0] * DIGITS
    exceptions = {}

    for x in range(256):
        d, q = decode(x, magics)

        for j in (1, 2, 3, 4):                                    # (1) exactness
            if q[j] != x // (3 ** j):
                bad.append(f"magic m{j}={magics[j]}: mulhi({x}) = {q[j]}, "
                           f"floor({x}/{3**j}) = {x // 3**j}")

        if min(d) < 0:                                            # (3) borrow
            bad.append(f"digit subtraction borrows at x={x}: {d}")

        for k in range(DIGITS):
            plane_max[k] = max(plane_max[k], d[k])

        if x <= 242:
            t, true = x, []                                       # (2) base-3
            for _ in range(DIGITS):
                true.append(t % 3)
                t //= 3
            if d != true:
                bad.append(f"x={x} decodes to {d}, base-3 is {true}")
            if any(v > 2 for v in d):                             # (4) range
                bad.append(f"x={x} is legal but leaves 0..2: {d}")
        elif any(v > 2 for v in d):
            exceptions[x] = d

    # (4) the exception set, asserted exactly and not merely counted
    if sorted(exceptions) != list(range(243, 256)):
        bad.append(f"bytes leaving 0..2 are {sorted(exceptions)}, expected 243..255")
    elif any(d[4] != 3 or max(d[:4]) > 2 for d in exceptions.values()):
        bad.append("a foreign byte leaves 0..2 by some digit other than d4=3")

    # (5) the accumulator bound, recomputed from the maxima just measured
    legal_max = [min(v, 2) for v in plane_max]
    legal = 2 * A_MAX * sum(legal_max)
    foreign = 2 * A_MAX * sum(plane_max)
    if fold * legal > INT16_MAX:
        bad.append(f"fold {fold} x {legal} = {fold * legal} exceeds {INT16_MAX}")
    if (fold + 1) * legal <= INT16_MAX:
        bad.append(f"fold could be {fold + 1}: {(fold+1) * legal} still fits")
    if fold * foreign <= INT16_MAX:
        bad.append(f"a block of foreign bytes would NOT overflow at fold {fold} "
                   f"({fold * foreign}) -- the precondition claim is wrong")
    return bad, plane_max, legal, foreign


def main() -> int:
    if SRC is None or HDR is None:
        print("  ternary_t5b sources not found in either layout -- skipping.")
        return 0

    magics, fold = read_constants()
    bad, plane_max, legal, foreign = evaluate(magics, fold)

    print("DECODE CLAIMS: the arithmetic itself, over all 256 byte values")
    print(f"  magics read from source  "
          f"{', '.join(f'/{3**j}={magics[j]}' for j in (1,2,3,4))}")
    print(f"  fold read from header    {fold}")
    print(f"  per-plane digit maxima   {plane_max}  (d0..d4, whole domain)")
    print(f"  lane bound per block     {legal} legal, {foreign} with foreign bytes")
    print(f"  fold headroom            {fold} x {legal} = {fold * legal} "
          f"<= {INT16_MAX} < {(fold + 1) * legal}")

    if bad:
        print(f"\n  Properties that do not hold ({len(bad)}):")
        for b in bad[:12]:
            print(f"    {b}")
        if len(bad) > 12:
            print(f"    ... and {len(bad) - 12} more")
        print("\nFAIL -- the decode does not have the properties the source and")
        print("  the paper claim for it.")
        return 1

    # The controls. Each must FAIL; a control that passes is not a control.
    if "--self-test" in sys.argv:
        controls = [
            ("m3 2428 -> 2427, exactness", {**magics, 3: 2427}, fold),
            ("m4 811 -> 813, first fails at x=242", {**magics, 4: 813}, fold),
            ("fold 12 -> 13, int16 headroom", magics, fold + 1),
        ]
        print("\n  negative controls")
        survived = []
        for name, mg, fd in controls:
            ctl, _, _, _ = evaluate(mg, fd)
            if ctl:
                print(f"    caught  {name}: {len(ctl)} broken; {ctl[0]}")
            else:
                print(f"    SURVIVED  {name}")
                survived.append(name)
        if survived:
            print("\nFAIL -- a control passed, so a pass here means nothing:")
            for s in survived:
                print(f"    {s}")
            return 1

    print("\nPASS -- exactness, losslessness, borrow-freedom, the digit range and")
    print("  its exception set, and the fold bound all hold by exhaustion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
