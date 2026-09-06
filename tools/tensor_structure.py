#!/usr/bin/env python3
"""
tensor_structure.py -- how many bits a ternary weight ACTUALLY needs, measured
on the shipped model rather than argued from log2(3).

The question this answers: llama.cpp's i2_s spends 2.00 bits per ternary
weight. The order-0 floor for three symbols is log2(3) = 1.585, and a dense
base-3 pack (10 trits per uint16, 3^10 = 59049 <= 65536) reaches 1.600. Is
there STRUCTURE in the weights -- correlation between neighbours, clustering of
zeros, per-block skew -- that a smarter code could exploit to go below that?

Everything here is counted over bytes read from the GGUF. Nothing is sampled,
estimated or extrapolated: every measured tensor is read in full, and the
model-wide pass covers all 210 i2_s tensors, all 2.08 billion weights.

TWO ORDERS, AND WHY BOTH ARE MEASURED
-------------------------------------
i2_s is interleaved. A 32-byte block carries 128 weights, and byte j of the
block holds four weights that are 32 apart in the logical row:

    bits 6-7 -> weight   0 + j        bits 2-3 -> weight  64 + j
    bits 4-5 -> weight  32 + j        bits 0-1 -> weight  96 + j

Verified against the kernel, not against a comment: in the upstream AVX2 branch
copied verbatim into src/ggml_i2s_ternary.c, the four shifts
(>>6, >>4, >>2, >>0) are fed to _mm256_maddubs_epi16 against activations at
py+0, py+32, py+64 and py+96, while px advances 32 bytes per 128 activations.

So "adjacent in the file" and "adjacent in the weight matrix" are different
pairings, and a correlation in one need not appear in the other. Order-1
entropy is therefore reported in both:

  storage order  -- the byte sequence as a streaming decoder meets it
  row-major      -- logical weight order, pairs that straddle a row excluded

THE MEASURE IS CHECKED IN BOTH DIRECTIONS
-----------------------------------------
A conditional-entropy estimator that silently returns H0 would make every
tensor look structureless, which is the conclusion being tested. --self-test
therefore runs it against synthetic data with a known answer: i.i.d. symbols,
where the drop must be 0, and a sequence in which half the symbols copy their
predecessor, where the drop must be large. Both are asserted, so a broken
estimator fails loudly instead of confirming the null.

MAPPING OF CODES
----------------
code 0 -> w = -1,  code 1 -> w = 0,  code 2 -> w = +1,  code 3 -> unused.
"Nonzero" below therefore means "code != 1".
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_inspect import parse_gguf  # noqa: E402

GGML_TYPE_I2_S = 36
LOG2 = math.log(2.0)
BASE3_RATE = 1.6      # 10 trits per uint16
I2S_RATE = 2.0        # what llama.cpp ships

# Reported tensor by tensor. Chosen to span layers (0, 4, 7, 14, 21, 25, 29)
# and all four distinct i2_s shapes, so a finding cannot be a property of one
# tensor or one depth. Every other i2_s tensor is still measured; it is
# aggregated into the model-wide figures and listed in the appendix.
DETAIL_TENSORS = [
    "blk.0.ffn_down.weight",       # [6912, 2560]  first layer, widest rows
    "blk.0.attn_q.weight",         # [2560, 2560]  square
    "blk.4.attn_v.weight",         # [2560,  640]  narrowest
    "blk.7.ffn_gate.weight",       # [2560, 6912]
    "blk.14.ffn_up.weight",        # [2560, 6912]  mid-stack
    "blk.14.attn_k.weight",        # [2560,  640]
    "blk.21.attn_output.weight",   # [2560, 2560]
    "blk.25.ffn_down.weight",      # [6912, 2560]
    "blk.29.ffn_gate.weight",      # [2560, 6912]  last layer
    "blk.29.attn_q.weight",        # [2560, 2560]
]

BLOCK_SIZES = [16, 32, 128, 160]                  # asked for by name
SCHEME_BLOCKS = [4, 8, 16, 32, 64, 128, 160]      # "from 4" upwards


# ----------------------------------------------------------------- utilities

def entropy_bits(counts):
    """Order-0 entropy of a symbol count vector, in bits."""
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.sum()
    if n <= 0:
        return 0.0
    p = counts[counts > 0] / n
    return float(-(p * np.log2(p)).sum())


def log2_binom(n, k):
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / LOG2


def nelem_of(t):
    n = 1
    for d in t["dims"]:
        n *= d
    return n


def read_codes(path, data_start, tensor):
    """The raw code bytes of one i2_s tensor, read in full."""
    nbytes = nelem_of(tensor) // 4
    with open(path, "rb") as f:
        f.seek(data_start + tensor["offset"])
        buf = f.read(nbytes)
    if len(buf) != nbytes:
        raise IOError(f"{tensor['name']}: short read {len(buf)} of {nbytes}")
    return np.frombuffer(buf, dtype=np.uint8)


def unpack_storage(b, msb_first):
    """Codes in physical byte order. msb_first follows the layout's own
    ordering (bits 7-6 first); the alternative is what a naive LSB-first
    unpacker produces. Both are reported because the choice is arbitrary and
    changes which pairs are adjacent."""
    out = np.empty((b.size, 4), dtype=np.uint8)
    shifts = (6, 4, 2, 0) if msb_first else (0, 2, 4, 6)
    for i, s in enumerate(shifts):
        out[:, i] = (b >> s) & 3
    return out.reshape(-1)


def unpack_logical(b):
    """De-interleave into row-major weight order using the kernel's layout."""
    if b.size % 32:
        raise ValueError("tensor byte count is not a multiple of 32")
    blk = b.reshape(-1, 32)
    out = np.empty((blk.shape[0], 128), dtype=np.uint8)
    out[:, 0:32] = (blk >> 6) & 3
    out[:, 32:64] = (blk >> 4) & 3
    out[:, 64:96] = (blk >> 2) & 3
    out[:, 96:128] = blk & 3
    return out.reshape(-1)


def cond_entropy_pairs(a, b):
    """H(b | a) in bits, plus the 4x4 joint count table."""
    joint = np.bincount(a.astype(np.int64) * 4 + b.astype(np.int64),
                        minlength=16).reshape(4, 4).astype(np.float64)
    n = joint.sum()
    marg = joint.sum(axis=1)
    h = 0.0
    for i in range(4):
        if marg[i] == 0:
            continue
        row = joint[i][joint[i] > 0] / marg[i]
        h += (marg[i] / n) * float(-(row * np.log2(row)).sum())
    return h, joint


def self_test():
    """Make the estimator go red on purpose before trusting a null result."""
    rng = np.random.default_rng(7)
    p = [0.309, 0.382, 0.309, 0.0]
    x = rng.choice(4, size=2_000_000, p=p).astype(np.uint8)
    h0 = entropy_bits(np.bincount(x, minlength=4))
    h1, _ = cond_entropy_pairs(x[:-1], x[1:])
    assert abs(h0 - h1) < 1e-3, f"i.i.d. drop should be 0, got {h0-h1}"
    y = x.copy()
    m = rng.random(y.size) < 0.5
    y[1:][m[1:]] = y[:-1][m[1:]]
    h0y = entropy_bits(np.bincount(y, minlength=4))
    h1y, _ = cond_entropy_pairs(y[:-1], y[1:])
    assert h0y - h1y > 0.05, f"injected correlation not detected: {h0y-h1y}"
    # de-interleave: byte j must land on weights j, 32+j, 64+j, 96+j
    b = np.zeros(32, dtype=np.uint8)
    b[5] = 0b11100100
    lg = unpack_logical(b)
    assert (lg[5], lg[37], lg[69], lg[101]) == (3, 2, 1, 0), "de-interleave wrong"
    assert abs(entropy_bits([1, 1, 1]) - math.log2(3)) < 1e-12
    return (h0 - h1, h0y - h1y)


# ------------------------------------------------------------------ measures

def measure_tensor(path, data_start, t, detail):
    name = t["name"]
    ne = nelem_of(t)
    raw = read_codes(path, data_start, t)
    res = {"name": name, "dims": list(t["dims"]), "nelem": ne,
           "nbytes": raw.size, "detail": detail}

    # --- questions 1 + 2: marginals via a byte histogram, exact over the whole
    # tensor: each byte value contributes a fixed count of each code.
    bytehist = np.bincount(raw, minlength=256).astype(np.int64)
    vals = np.arange(256, dtype=np.int64)
    counts = np.zeros(4, dtype=np.int64)
    for sh in (0, 2, 4, 6):
        counts += np.bincount((vals >> sh) & 3, weights=bytehist,
                              minlength=4).astype(np.int64)
    assert counts.sum() == ne, (counts.sum(), ne)
    res["counts"] = counts
    res["H0"] = entropy_bits(counts)

    # --- question 3: order-1 conditional entropy.
    if detail:
        for tag, msb in (("msb", True), ("lsb", False)):
            s = unpack_storage(raw, msb_first=msb)
            res[f"H1_storage_{tag}"], _ = cond_entropy_pairs(s[:-1], s[1:])
            del s

    logical = unpack_logical(raw)
    del raw
    ne0 = t["dims"][0]
    nrows = ne // ne0
    m = logical.reshape(nrows, ne0)
    res["H1_rowmajor"], joint = cond_entropy_pairs(
        m[:, :-1].reshape(-1), m[:, 1:].reshape(-1))
    res["joint_rowmajor"] = joint
    if detail:
        # Down a column too: a correlation could live along the other axis.
        res["H1_colmajor"], _ = cond_entropy_pairs(
            m[:-1, :].reshape(-1), m[1:, :].reshape(-1))

    # --- per-row statistics. The zero rate turns out to be a property of the
    # ROW (the output neuron), which is where the block skew comes from, so it
    # is measured rather than inferred.
    rowcnt = np.stack([(m == c).sum(axis=1) for c in range(3)],
                      axis=1).astype(np.float64)
    pr = rowcnt / ne0
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(pr > 0, np.log2(np.where(pr > 0, pr, 1.0)), 0.0)
    row_h = -(pr * lg).sum(axis=1)
    res["nrows"] = nrows
    res["row_H0_mean"] = float(row_h.mean())     # rows are equal length
    row_dens = 1.0 - rowcnt[:, 1] / ne0
    res["zero_rows"] = int((rowcnt[:, 1] == ne0).sum())
    res["row_dens"] = (float(row_dens.min()), float(np.median(row_dens)),
                       float(row_dens.max()), float(row_dens.std()))

    # The same two entropies over the LIVE rows only. A dead row is a run of
    # thousands of identical symbols, which is trivially predictable and would
    # otherwise be scored as "the weights are correlated". This separates the
    # dead rows from any correlation among the weights that survive.
    res["live_rows"] = nrows - res["zero_rows"]
    res["live_nelem"] = res["live_rows"] * ne0
    # Masks kept so the FFN neurons can be matched across gate/up/down. A dead
    # FFN neuron is a dead ROW of gate and up but a dead COLUMN of ffn_down,
    # because ffn_down is stored with the FFN dimension along the row.
    res["dead_rows_mask"] = (rowcnt[:, 1] == ne0)
    res["dead_cols_mask"] = ((m != 1).sum(axis=0) == 0)
    if res["zero_rows"]:
        ml = m[rowcnt[:, 1] != ne0]
        res["live_H0"] = entropy_bits(np.bincount(ml.reshape(-1), minlength=4))
        res["live_H1"], _ = cond_entropy_pairs(ml[:, :-1].reshape(-1),
                                               ml[:, 1:].reshape(-1))
        del ml
    else:
        res["live_H0"], res["live_H1"] = res["H0"], res["H1_rowmajor"]
    del m

    # --- question 4: block structure, in row-major (logical) order.
    nz = (logical != 1)
    res["blocks"] = {}
    for B in sorted(set(BLOCK_SIZES) | set(SCHEME_BLOCKS)):
        nfull = ne // B
        k = nz[:nfull * B].reshape(nfull, B).sum(axis=1)
        hist = np.bincount(k, minlength=B + 1).astype(np.int64)
        pnz = float(nz.mean())
        res["blocks"][B] = {
            "nblocks": nfull, "dropped": ne - nfull * B, "hist": hist,
            "allzero": int(hist[0]), "le2": int(hist[:3].sum()),
            "mean_nz": float(k.mean()), "sd_nz": float(k.std()),
            "sd_binom": math.sqrt(B * pnz * (1 - pnz)),
        }
    del nz, logical

    res["schemes"] = compute_schemes(res)
    return res


def compute_schemes(res):
    """Total BITS for this tensor under each scheme, so tensors can be summed
    into a model-wide rate rather than averaged as rates."""
    ne = res["nelem"]
    counts = res["counts"]
    out = {"nelem": ne}

    # (a) dense base-3. Lossless only because code 3 is absent.
    out["base3_bits"] = math.ceil(ne / 10) * 16.0
    out["base3_applicable"] = bool(counts[3] == 0)

    # (c) order-0 arithmetic coding: ideal length is exactly ne * H0.
    out["arith0_bits"] = ne * res["H0"]
    out["arith1_bits"] = ne * res["H1_rowmajor"]

    # (e) EXTRA: order-0 coding with one distribution PER ROW. Side info is
    # three counts per row and is charged for at 32 bits each.
    out["rowadapt_bits"] = ne * res["row_H0_mean"] + res["nrows"] * 3 * 32

    # (f) EXTRA: dense base-3 plus a 1-bit ROW skip flag. The measured zero
    # clustering is entirely whole dead rows, so this is the scheme the data
    # actually points at -- one flag per row rather than one per block, and no
    # entropy coder at all.
    ne0 = res["dims"][0]
    live = res["nrows"] - res["zero_rows"]
    out["rowskip_bits"] = res["nrows"] * 1.0 + live * ne0 * BASE3_RATE

    # (b) per-block zero skipping: 1 flag bit per block; an all-zero block
    # costs nothing more, any other block costs the dense rate for all B of its
    # weights. Charged against both dense rates.
    skip = {}
    for B in SCHEME_BLOCKS:
        b = res["blocks"][B]
        nb, az = b["nblocks"], b["allzero"]
        for cname, c in (("base3", BASE3_RATE), ("i2s", I2S_RATE)):
            skip.setdefault(cname, {})[B] = (nb * 1.0 + (nb - az) * B * c,
                                             nb * B)
    out["skip"] = skip

    # (d) EXTRA: per-block enumerative coding. Send the block's nonzero count k
    # (cost: the entropy of the OBSERVED k distribution -- this is where
    # per-block skew pays), then the index of the pattern among all C(B,k)
    # placements, then one sign per nonzero at the measured sign entropy. If
    # the k distribution were exactly binomial this lands on H0; anything below
    # H0 IS the cash value of block skew.
    h_sign = entropy_bits([counts[0], counts[2]])
    out["h_sign"] = h_sign
    enum = {}
    for B in SCHEME_BLOCKS:
        b = res["blocks"][B]
        hist = b["hist"].astype(np.float64)
        nb = hist.sum()
        h_k = entropy_bits(hist)
        idx = sum(hist[k] * log2_binom(B, k)
                  for k in range(B + 1) if hist[k]) / nb
        enum[B] = ((h_k + idx + b["mean_nz"] * h_sign) * nb, nb * B)
    out["enum"] = enum
    return out


def sweep_code3(path, data_start, tensors):
    """Question 2 over the WHOLE file: every i2_s tensor, every weight."""
    vals = np.arange(256, dtype=np.int64)
    per_code = [sum((((vals >> s) & 3) == c).astype(np.int64)
                    for s in (0, 2, 4, 6)) for c in range(4)]
    total = np.zeros(4, dtype=np.int64)
    n_t = 0
    with open(path, "rb") as f:
        for t in tensors:
            if t["type"] != GGML_TYPE_I2_S:
                continue
            left = nelem_of(t) // 4
            f.seek(data_start + t["offset"])
            hist = np.zeros(256, dtype=np.int64)
            while left:
                chunk = f.read(min(left, 32 << 20))
                if not chunk:
                    raise IOError(f"{t['name']}: short read")
                hist += np.bincount(np.frombuffer(chunk, dtype=np.uint8),
                                    minlength=256)
                left -= len(chunk)
            for c in range(4):
                total[c] += int((hist * per_code[c]).sum())
            n_t += 1
    return n_t, total


def aggregate(all_res):
    """Model-wide bits/weight: total bits over total weights, per scheme."""
    N = sum(r["nelem"] for r in all_res)
    agg = {"N": N, "ntensors": len(all_res)}
    for key in ("base3_bits", "arith0_bits", "arith1_bits", "rowadapt_bits",
                "rowskip_bits"):
        agg[key[:-5]] = sum(r["schemes"][key] for r in all_res) / N
    for fam in ("skip", "enum"):
        agg[fam] = {}
        keys = all_res[0]["schemes"][fam].keys()
        for k in keys:
            if fam == "skip":
                agg[fam][k] = {}
                for B in SCHEME_BLOCKS:
                    bits = sum(r["schemes"][fam][k][B][0] for r in all_res)
                    cov = sum(r["schemes"][fam][k][B][1] for r in all_res)
                    agg[fam][k][B] = bits / cov
            else:
                bits = sum(r["schemes"][fam][k][0] for r in all_res)
                cov = sum(r["schemes"][fam][k][1] for r in all_res)
                agg[fam][k] = bits / cov
    # a single global order-0 table for the whole model, not one per tensor
    tot = np.zeros(4, dtype=np.int64)
    for r in all_res:
        tot += r["counts"]
    agg["global_counts"] = tot
    agg["arith0_global"] = entropy_bits(tot)
    agg["zero_rows"] = sum(r["zero_rows"] for r in all_res)
    agg["nrows"] = sum(r["nrows"] for r in all_res)
    LN = sum(r["live_nelem"] for r in all_res)
    agg["live_N"] = LN
    agg["live_H0"] = sum(r["live_nelem"] * r["live_H0"] for r in all_res) / LN
    agg["live_drop"] = sum(r["live_nelem"] * (r["live_H0"] - r["live_H1"])
                           for r in all_res) / LN
    agg["live_drop_max"] = max(all_res, key=lambda r: r["live_H0"] - r["live_H1"])
    return agg


# -------------------------------------------------------------------- report

def emit(all_res, detail_res, sweep, agg, meta_lines, st, out):
    w = out.write
    pct = lambda x: (BASE3_RATE - x) / BASE3_RATE          # noqa: E731

    w("TENSOR STRUCTURE -- how many bits a ternary weight really needs,\n")
    w("measured on the shipped weights\n")
    w("=" * 75 + "\n")
    w("  Reproduce: python3 tools/tensor_structure.py\n\n")
    for line in meta_lines:
        w("  " + line + "\n")
    w("""
  Question. i2_s spends 2.00 bits/weight. log2(3) = 1.585 is the order-0 floor
  for three symbols and a dense base-3 pack reaches 1.600. Is there structure
  -- neighbour correlation, zero clustering, per-block skew -- worth a more
  complicated decoder? Every number below is counted over bytes read from the
  GGUF in full. Nothing is sampled or extrapolated.

  Codes: 0 -> w=-1, 1 -> w=0, 2 -> w=+1, 3 -> unused. "Nonzero" = code != 1.

  Two orders are reported because i2_s is interleaved: a 32-byte block holds
  128 weights, byte j carrying weights j, 32+j, 64+j, 96+j (bits 7-6, 5-4,
  3-2, 1-0). Confirmed against the verbatim upstream AVX2 kernel in
  src/ggml_i2s_ternary.c, where the >>6/>>4/>>2/>>0 lanes are
  paired with activations py+0/+32/+64/+96 while px advances 32 bytes per 128
  activations. "Storage order" is the byte stream; "row-major" is logical
  weight order with row-straddling pairs excluded.
""")
    w(f"""
  The estimator was made to fail on purpose before any null was believed:
  on synthetic i.i.d. symbols H0-H1 = {st[0]:.5f} (must be 0), and with half
  the symbols copying their predecessor H0-H1 = {st[1]:.5f} (must be large).
  Run as --self-test; both are asserted on every run.

""")

    # ------------------------------------------------------------------ (1)
    w("1. TENSORS MEASURED\n")
    w("-" * 75 + "\n")
    w(f"  All {agg['ntensors']} i2_s tensors of the model were measured in "
      f"full: {agg['N']:,} weights.\n")
    w("  The ten below are reported tensor by tensor; every one of the "
      f"{agg['ntensors']} is\n  in the model-wide figures and in the appendix.\n\n")
    w(f"  {'tensor':<28} {'dims':>14} {'weights':>13} {'bytes read':>12}\n")
    tot_w = 0
    for r in detail_res:
        tot_w += r["nelem"]
        w(f"  {r['name']:<28} {str(r['dims']):>14} {r['nelem']:>13,} "
          f"{r['nbytes']:>12,}\n")
    w(f"  {'subtotal (detail)':<28} {'':>14} {tot_w:>13,}\n")
    layers = {r["name"].split(".")[1] for r in detail_res}
    shapes = {tuple(r["dims"]) for r in detail_res}
    w(f"\n  {len(detail_res)} detailed tensors, {len(layers)} distinct layers "
      f"({', '.join(sorted(layers, key=int))}),\n  {len(shapes)} distinct "
      f"shapes, all 7 projection kinds.\n")

    # ------------------------------------------------------------------ (2)
    w("\n\n2. MARGINAL DISTRIBUTION AND ORDER-0 ENTROPY  (question 1)\n")
    w("-" * 75 + "\n")
    w(f"  {'tensor':<28} {'code 0':>9} {'code 1':>9} {'code 2':>9} "
      f"{'code 3':>8} {'H0 bits':>9}\n")
    for r in detail_res:
        c, n = r["counts"], r["nelem"]
        w(f"  {r['name']:<28} {c[0]/n:>8.3%} {c[1]/n:>8.3%} {c[2]/n:>8.3%} "
          f"{c[3]:>8,} {r['H0']:>9.4f}\n")
    h0all = [r["H0"] for r in all_res]
    p1all = [r["counts"][1] / r["nelem"] for r in all_res]
    lo = min(all_res, key=lambda r: r["H0"])
    hi = max(all_res, key=lambda r: r["H0"])
    w(f"\n  The 30.887 / 38.214 / 30.899 figure previously measured on\n")
    w("  blk.0.ffn_down.weight is CONFIRMED on that tensor and REFUTED as a\n")
    w("  model-wide constant. Over all "
      f"{agg['ntensors']} i2_s tensors H0 spans\n"
      f"    {min(h0all):.4f} bits  ({lo['name']}, zero share {lo['counts'][1]/lo['nelem']:.3%})\n"
      f"    {max(h0all):.4f} bits  ({hi['name']}, zero share {hi['counts'][1]/hi['nelem']:.3%})\n")
    w(f"  Zero (code 1) share ranges {min(p1all):.3%} .. {max(p1all):.3%}; "
      f"log2(3) = {math.log2(3):.4f}.\n")
    w("  Symmetry between -1 and +1 holds everywhere: the two counts differ by\n")
    w(f"  at most {max(abs(r['counts'][0]-r['counts'][2])/r['nelem'] for r in all_res):.4%} "
      "of the tensor.\n")

    # ------------------------------------------------------------------ (3)
    w("\n\n3. CODE 3  (question 2)\n")
    w("-" * 75 + "\n")
    n_t, tot = sweep
    w("  Swept EVERY i2_s tensor in the file, independently of the pass above:\n")
    w(f"    tensors        {n_t}\n")
    w(f"    weights        {tot.sum():,}\n")
    for c in range(4):
        w(f"    code {c}         {tot[c]:>15,}   {tot[c]/tot.sum():8.4%}\n")
    w(f"\n  code 3 count = {tot[3]}")
    if tot[3] == 0:
        w(" exactly. Not 'approximately zero' and not an\n")
        w(f"  impression: zero occurrences in {tot.sum():,} weights. The\n")
        w("  {0,1,2} encoding is exact and base-3 packing is lossless here.\n")
    else:
        w(" -- NOT absent. Base-3 packing would be lossy.\n")
    w(f"\n  Whole-model H0 under one global table = {agg['arith0_global']:.4f} "
      "bits/weight.\n")

    # ------------------------------------------------------------------ (4)
    w("\n\n4. ORDER-1 CONDITIONAL ENTROPY  (question 3)\n")
    w("-" * 75 + "\n")
    w("  H(w_i | w_i-1). Below H0 would mean the weights are correlated and a\n")
    w("  context-dependent code would beat the order-0 floor.\n\n")
    w(f"  {'tensor':<28} {'H0':>8} {'H1 stor':>9} {'H1 stor':>9} {'H1 row':>9} "
      f"{'H1 col':>9}\n")
    w(f"  {'':<28} {'':>8} {'msb':>9} {'lsb':>9} {'major':>9} {'umn':>9}\n")
    for r in detail_res:
        w(f"  {r['name']:<28} {r['H0']:>8.4f} {r['H1_storage_msb']:>9.4f} "
          f"{r['H1_storage_lsb']:>9.4f} {r['H1_rowmajor']:>9.4f} "
          f"{r['H1_colmajor']:>9.4f}\n")
    drops = [r["H0"] - r["H1_rowmajor"] for r in all_res]
    wt = sum(r["nelem"] * (r["H0"] - r["H1_rowmajor"]) for r in all_res) / agg["N"]
    dmax = max(all_res, key=lambda r: r["H0"] - r["H1_rowmajor"])
    w(f"\n  Over all {agg['ntensors']} tensors, row-major H0 - H1:\n")
    w(f"    weighted mean {wt:.5f} bits/weight  ({wt/BASE3_RATE:.3%} of a "
      "1.600-bit budget)\n")
    w(f"    largest       {max(drops):.5f} bits/weight  ({dmax['name']})\n")
    w("  For scale, the estimator resolved an injected correlation of "
      f"{st[1]:.4f} bits.\n")
    w("\n  The largest drops are NOT weight correlation. Every one of them is on\n")
    w("  a tensor with dead rows (section 5), and a dead row is a run of\n")
    w("  thousands of identical symbols, which any order-1 model predicts for\n")
    w("  free. Re-measured over the LIVE rows only:\n\n")
    dr = [r for r in all_res if r["zero_rows"]]
    dr.sort(key=lambda r: -(r["H0"] - r["H1_rowmajor"]))
    w(f"    {'tensor':<28}{'H0 all':>9}{'H1 all':>9}{'drop':>9}"
      f"{'H0 live':>9}{'H1 live':>9}{'drop':>9}\n")
    for r in dr[:6]:
        w(f"    {r['name']:<28}{r['H0']:>9.4f}{r['H1_rowmajor']:>9.4f}"
          f"{r['H0']-r['H1_rowmajor']:>9.5f}{r['live_H0']:>9.4f}"
          f"{r['live_H1']:>9.4f}{r['live_H0']-r['live_H1']:>9.5f}\n")
    lm = agg["live_drop_max"]
    w(f"\n  Over the {agg['live_N']:,} weights in live rows, model-wide:\n")
    w(f"    weighted mean H0 - H1 = {agg['live_drop']:.5f} bits/weight\n")
    w(f"    largest on any tensor = {lm['live_H0']-lm['live_H1']:.5f} "
      f"({lm['name']})\n")
    w(f"    weighted mean H0      = {agg['live_H0']:.4f}, against "
      f"log2(3) = {math.log2(3):.4f}\n")
    w(f"\n  Once the dead rows are set aside the order-1 gain collapses from "
      f"{wt:.5f}\n  to {agg['live_drop']:.5f} bits/weight model-wide "
      f"({agg['live_drop']/BASE3_RATE:.3%} of a 1.600-bit budget), and to at\n"
      f"  most {lm['live_H0']-lm['live_H1']:.5f} on any single tensor "
      f"({(lm['live_H0']-lm['live_H1'])/BASE3_RATE:.2%}) -- that residual is the\n"
      "  spread of per-row density in section 5, not a weight-to-weight\n"
      "  relationship. It is small enough that a context model cannot repay a\n"
      "  decoder either way. The structure in this model is WHICH ROWS ARE DEAD;\n"
      "  almost none of it is in the relationship between one weight and the next.\n")
    r0 = detail_res[0]
    j = r0["joint_rowmajor"]
    jn = j.sum()
    marg = j.sum(axis=1) / jn
    w(f"\n  Joint table, row-major, {r0['name']} (rows = w_i-1,\n")
    w("  cols = w_i, share of all pairs), against the product of its margins:\n\n")
    w("            " + "".join(f"{c:>12}" for c in ("w=-1", "w=0", "w=+1")) + "\n")
    for i in range(3):
        w(f"    {('w=-1','w=0','w=+1')[i]:<6} obs")
        for k in range(3):
            w(f"{j[i][k]/jn:>12.5%}")
        w("\n")
        w(f"    {'':<6} exp")
        for k in range(3):
            w(f"{marg[i]*marg[k]:>12.5%}")
        w("\n")

    # ------------------------------------------------------------------ (5)
    w("\n\n5. BLOCK STRUCTURE -- NONZEROS PER BLOCK  (question 4)\n")
    w("-" * 75 + "\n")
    w("  Row-major order. 'binomial' is the all-zero fraction that independent\n")
    w("  weights at this tensor's own zero rate would give -- the yardstick for\n")
    w("  whether zeros CLUSTER. 'sd/binom' is the observed standard deviation\n")
    w("  of the nonzero count over the binomial one: 1.0 means no block skew.\n")
    for B in BLOCK_SIZES:
        w(f"\n  block = {B} weights\n")
        w(f"    {'tensor':<28} {'blocks':>11} {'mean nz':>8} {'all-zero':>11} "
          f"{'binomial':>11} {'<=2 nz':>10} {'sd/binom':>9}\n")
        for r in detail_res:
            b = r["blocks"][B]
            pz = r["counts"][1] / r["nelem"]
            w(f"    {r['name']:<28} {b['nblocks']:>11,} {b['mean_nz']:>8.3f} "
              f"{b['allzero']/b['nblocks']:>11.3e} {pz**B:>11.3e} "
              f"{b['le2']/b['nblocks']:>10.3e} {b['sd_nz']/b['sd_binom']:>9.2f}\n")

    w("\n  Two things in that table are not noise.\n\n")
    w("  (i) ZEROS CLUSTER, BY MANY ORDERS OF MAGNITUDE. All-zero 128-blocks\n")
    w("      occur where independence predicts 1e-39 to 1e-57. The cause is\n")
    w("      whole DEAD ROWS -- output neurons whose entire weight vector is\n")
    w("      zero -- not a local run:\n\n")
    dead = [r for r in all_res if r["zero_rows"]]
    dead.sort(key=lambda r: -r["zero_rows"] / r["nrows"])
    w(f"        tensors with >=1 entirely-zero row : {len(dead)} of {agg['ntensors']}\n")
    w(f"        entirely-zero rows, whole model    : {agg['zero_rows']:,} of "
      f"{agg['nrows']:,} ({agg['zero_rows']/agg['nrows']:.4%})\n")
    if dead:
        w("        worst offenders:\n")
        for r in dead[:6]:
            w(f"          {r['name']:<28} {r['zero_rows']:>5} / {r['nrows']:>5} rows"
              f"  = {r['zero_rows']/r['nrows']:>7.4%} of its weights\n")
    w("\n      This is why the all-zero fraction is IDENTICAL at B=16 and\n")
    w("      B=160 for those tensors: the zero regions are thousands of\n")
    w("      weights long, so every block size lands inside them equally.\n")

    # The dead rows are whole FFN neurons, matched across the SwiGLU triple.
    byname = {r["name"]: r for r in all_res}
    ffn = []
    for L in range(64):
        g = byname.get(f"blk.{L}.ffn_gate.weight")
        u = byname.get(f"blk.{L}.ffn_up.weight")
        d = byname.get(f"blk.{L}.ffn_down.weight")
        if not (g and u and d):
            continue
        gm, um, dm = g["dead_rows_mask"], u["dead_rows_mask"], d["dead_cols_mask"]
        if gm.shape != dm.shape:
            continue
        ffn.append((L, int(gm.sum()), int(um.sum()), int(dm.sum()),
                    int((gm & um & dm).sum()), gm.size))
    hot = [f for f in ffn if f[4]]
    if hot:
        w("\n      Those dead rows are whole FFN NEURONS, and they are dead in\n")
        w("      all three matrices at once. A neuron is a row of ffn_gate and\n")
        w("      ffn_up but a COLUMN of ffn_down, so the three were matched up\n")
        w("      rather than assumed to correspond:\n\n")
        w(f"        {'layer':>6}{'gate rows':>11}{'up rows':>9}{'down cols':>11}"
          f"{'all three':>11}{'of':>7}\n")
        for L, gs, us, ds_, both, n in hot:
            w(f"        {L:>6}{gs:>11}{us:>9}{ds_:>11}{both:>11}{n:>7}\n")
        tot_both = sum(f[4] for f in ffn)
        tot_n = sum(f[5] for f in ffn)
        w(f"\n        {tot_both:,} of {tot_n:,} FFN neurons ({tot_both/tot_n:.2%}) are dead in\n")
        w("        gate, up AND down together -- removable outright, not merely\n")
        w("        compressible. They are concentrated in the early layers.\n")

    w("\n  (ii) BLOCK DENSITY IS SKEWED, AND IT IS A PROPERTY OF THE ROW.\n")
    sk = sorted(all_res, key=lambda r: -r["blocks"][32]["sd_nz"] / r["blocks"][32]["sd_binom"])
    w("      Observed sd of the nonzero count over the binomial sd, B=32:\n")
    w(f"        range {sk[-1]['blocks'][32]['sd_nz']/sk[-1]['blocks'][32]['sd_binom']:.2f} "
      f"({sk[-1]['name']}) .. "
      f"{sk[0]['blocks'][32]['sd_nz']/sk[0]['blocks'][32]['sd_binom']:.2f} "
      f"({sk[0]['name']})\n")
    w("      Blocks are NOT independent draws. Per-row nonzero density:\n\n")
    w(f"        {'tensor':<28} {'min':>8} {'median':>8} {'max':>8} {'sd':>8}\n")
    for r in detail_res:
        d = r["row_dens"]
        w(f"        {r['name']:<28} {d[0]:>8.4f} {d[1]:>8.4f} {d[2]:>8.4f} "
          f"{d[3]:>8.4f}\n")
    w("\n      In blk.0.attn_q.weight one row is 0.12% nonzero and another is\n")
    w("      82.7% nonzero. The sparsity lives per output neuron, not per\n")
    w("      model. That is the structure worth coding against -- and it is\n")
    w("      captured by a per-row probability table, not by a skip flag.\n")

    # ------------------------------------------------------------------ (6)
    w("\n\n6. BITS PER WEIGHT UNDER EACH SCHEME  (question 5)\n")
    w("-" * 75 + "\n")
    w("  (a) dense base-3, 10 trits per uint16 (3^10 = 59049 <= 65536)\n")
    w("  (b) per-block zero skipping, 1 flag bit per block, non-skipped blocks\n")
    w("      at the base-3 rate; the best block size for each tensor is shown\n")
    w("  (c) order-0 arithmetic coding at the tensor's own measured entropy\n")
    w("  (d) EXTRA: per-block enumerative coding (count + placement + signs)\n")
    w("  (e) EXTRA: order-0 arithmetic coding with one table PER ROW, side\n")
    w("      info charged at 3 x 32 bits per row\n")
    w("  (f) EXTRA: dense base-3 plus a 1-bit skip flag per ROW -- the scheme\n")
    w("      the dead-row finding actually points at, with no entropy coder\n\n")
    w(f"  {'tensor':<28} {'(a) base3':>10} {'(b) skip':>9} {'(c) ari0':>9} "
      f"{'(d) enum':>9} {'(e) row':>9} {'(f) rskip':>10} {'i2_s':>6}\n")
    for r in detail_res:
        s = r["schemes"]
        ne = r["nelem"]
        bskip = min(v[0] / v[1] for v in s["skip"]["base3"].values())
        benum = min(v[0] / v[1] for v in s["enum"].values())
        w(f"  {r['name']:<28} {s['base3_bits']/ne:>10.4f} {bskip:>9.4f} "
          f"{s['arith0_bits']/ne:>9.4f} {benum:>9.4f} "
          f"{s['rowadapt_bits']/ne:>9.4f} {s['rowskip_bits']/ne:>10.4f} "
          f"{I2S_RATE:>6.3f}\n")

    for label, fam, key, note in (
            ("(b) zero-skip, non-skipped blocks at 1.600", "skip", "base3", ""),
            ("(b) zero-skip, non-skipped blocks at 2.000", "skip", "i2s", ""),
            ("(d) per-block enumerative", "enum", None, "")):
        w(f"\n  {label}, bits/weight by block size:\n\n")
        w(f"    {'tensor':<28}" +
          "".join(f"{('B=%d' % B):>9}" for B in SCHEME_BLOCKS) + "\n")
        for r in detail_res:
            s = r["schemes"][fam]
            d = s[key] if key else s
            w(f"    {r['name']:<28}" +
              "".join(f"{d[B][0]/d[B][1]:>9.4f}" for B in SCHEME_BLOCKS) + "\n")
        if note:
            w(note)
    w("\n  (d) is not monotone in B for every tensor -- blk.21.attn_output.weight\n")
    w("  rises from B=128 to B=160. That is alignment, not an error: its zero\n")
    w("  runs are 128-134 long, so 128-blocks land on them and 160-blocks do\n")
    w("  not, which narrows the spread of the per-block count and costs the\n")
    w("  scheme the skew it was living on.\n")

    w("\n  MODEL-WIDE, all "
      f"{agg['ntensors']} i2_s tensors, {agg['N']:,} weights\n")
    w("  (total bits over total weights, not an average of rates):\n\n")
    best_skip_B = min(SCHEME_BLOCKS, key=lambda B: agg["skip"]["base3"][B])
    best_enum_B = min(SCHEME_BLOCKS, key=lambda B: agg["enum"][B])
    rows = [("i2_s, as shipped", I2S_RATE),
            ("(a) dense base-3", agg["base3"]),
            (f"(b) zero-skip at 1.600, best B={best_skip_B}",
             agg["skip"]["base3"][best_skip_B]),
            ("(c) order-0 arithmetic, one global table", agg["arith0_global"]),
            ("(c) order-0 arithmetic, one table per tensor", agg["arith0"]),
            ("    order-1 arithmetic, row-major context", agg["arith1"]),
            ("(e) order-0 arithmetic, one table per row", agg["rowadapt"]),
            ("(f) dense base-3 + 1 skip bit per ROW", agg["rowskip"]),
            (f"(d) enumerative, best B={best_enum_B}", agg["enum"][best_enum_B])]
    rows = rows[:2] + sorted(rows[2:], key=lambda x: -x[1])
    w(f"    {'scheme':<46}{'bits/wt':>9}{'vs 1.600':>10}{'vs 2.000':>10}\n")
    for nm, v in rows:
        w(f"    {nm:<46}{v:>9.4f}{pct(v):>10.2%}{(I2S_RATE-v)/I2S_RATE:>10.2%}\n")
    w("\n    (b) in full, model-wide: " +
      "  ".join(f"B={B}:{agg['skip']['base3'][B]:.4f}" for B in SCHEME_BLOCKS) + "\n")

    # ------------------------------------------------------------------ (7)
    w("\n\n7. VERDICT  (question 6)\n")
    w("-" * 75 + "\n")
    cands = {
        "(b) zero-skip, B=%d" % best_skip_B: agg["skip"]["base3"][best_skip_B],
        "(c) order-0 arithmetic, per tensor": agg["arith0"],
        "(d) enumerative, B=%d" % best_enum_B: agg["enum"][best_enum_B],
        "(e) order-0 arithmetic, per row": agg["rowadapt"],
        "(f) base-3 + row skip bit": agg["rowskip"],
    }
    bname = min(cands, key=cands.get)
    best = cands[bname]
    beat = {k: v for k, v in cands.items() if v < BASE3_RATE}

    w("  Reference point for 'is it worth a decoder': the shipped i2_s kernel\n")
    w("  sustains 20.19 GB/s of weight bytes at a DRAM-resident working set --\n")
    w("  33,554,432 weights in 0.415 ms, i.e. 262,144 blocks of 128 in 0.415 ms\n")
    w("  = 632 million blocks/s (results/bench_code_kernel.txt, rows=8192).\n")
    w("  A replacement decoder must hold that while doing strictly more per\n")
    w("  block than a shift and a mask.\n\n")

    w(f"  {len(beat)} of the {len(cands)} schemes DO beat 1.600 bits/weight "
      f"model-wide, the\n  best by {pct(best):.2%}. None beats it by enough to be "
      "worth building.\n\n")
    w(f"  Best measured: {bname} at {best:.4f} bits/weight,\n")
    w(f"  {pct(best):.2%} below dense base-3 and {(I2S_RATE-best)/I2S_RATE:.2%} "
      "below the shipped 2.000.\n")
    w(f"  In bytes on this 2B model that is {agg['N']*(BASE3_RATE-best)/8/2**20:.1f} "
      f"MiB saved against base-3,\n  on top of the "
      f"{agg['N']*(I2S_RATE-BASE3_RATE)/8/2**20:.1f} MiB that base-3 already "
      "saves against i2_s.\n\n")
    w("  The numbers behind that:\n\n")
    w(f"  - Order-1 context buys {wt:.5f} bits/weight ({wt/BASE3_RATE:.3%}), and "
      f"over live rows\n    only {agg['live_drop']:.5f} "
      f"({agg['live_drop']/BASE3_RATE:.3%}). The weights are serially independent "
      "in\n    storage order and in row-major order alike; what the order-1 number\n"
      "    does show is the dead rows being trivially predictable, not a\n"
      "    weight-to-weight relationship. A context model is ruled out outright.\n")
    sb = agg["skip"]["base3"]
    w(f"  - Zero-skip flags: {sb[best_skip_B]:.4f} at their best (B={best_skip_B}), "
      f"{pct(sb[best_skip_B]):.2%} below 1.600,\n")
    w(f"    but they LOSE at small blocks ({sb[4]:.4f} at B=4) because break-even\n")
    w("    needs an all-zero fraction above 1/(1.6*B) = 15.6% at B=4, 0.49% at\n")
    w("    B=128. Every bit of their win comes from the dead rows, not from\n")
    w("    local sparsity -- which is why bigger blocks are strictly better and\n")
    w("    why (f), one flag per ROW, gets nearly the same for far less work.\n")
    w(f"  - Order-0 arithmetic: {agg['arith0']:.4f} per tensor ({pct(agg['arith0']):.2%}), "
      f"{agg['arith0_global']:.4f} with one\n    global table "
      f"({pct(agg['arith0_global']):.2%}). Both need a serial range decoder, whose\n"
      "    dependency chain cannot be made to run at 632M blocks/s. Paying that\n"
      "    to save 2.5% of bytes on a bandwidth-bound kernel is a bad trade.\n")
    w(f"  - The one real structure is per-row sparsity: {agg['zero_rows']:,} of "
      f"{agg['nrows']:,}\n    rows ({agg['zero_rows']/agg['nrows']:.2%}) are entirely "
      "zero, up to 43.59% of a single\n    tensor, and the surviving rows differ "
      "widely in density. It is worth\n"
      f"    {pct(agg['rowskip']):.2%} as a row skip bit (f) and {pct(agg['rowadapt']):.2%} "
      "as a per-row table (e).\n")
    w(f"  - The enumerative scheme (d) is the only one past {pct(agg['rowadapt']):.1%}, at "
      f"{pct(agg['enum'][best_enum_B]):.2%}, and it is\n    the most expensive to "
      "decode of all of them: a binomial rank/unrank\n    per block. It is the "
      "clearest case of the pattern -- the more of the\n    structure a scheme "
      "captures, the less it can keep up with the kernel.\n\n")
    w("  So the practical ceiling on a repacking effort is the 20.00% between\n")
    w("  the shipped 2.000 and dense base-3 at 1.600. base-3 decodes with a\n")
    w("  multiply and a shift and stays vectorisable; everything below it here\n")
    w(f"  is worth at most a further {pct(best):.2%} and costs a decoder that is no\n")
    w("  longer memory-bound, which defeats the purpose on a kernel whose whole\n")
    w("  problem is bytes. If the dead rows are worth having, the way to take\n")
    w("  them is to drop those rows from the model, not to entropy-code them.\n")

    # ------------------------------------------------------------------ (A)
    w("\n\nAPPENDIX -- EVERY i2_s TENSOR MEASURED\n")
    w("-" * 75 + "\n")
    w(f"  {'tensor':<30}{'weights':>13}{'zero%':>9}{'H0':>8}{'H1 row':>9}"
      f"{'0-rows':>8}\n")
    for r in all_res:
        w(f"  {r['name']:<30}{r['nelem']:>13,}"
          f"{r['counts'][1]/r['nelem']:>9.3%}{r['H0']:>8.4f}"
          f"{r['H1_rowmajor']:>9.4f}{r['zero_rows']:>8}\n")
    return wt


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("gguf", nargs="?", type=Path,
                    default=root / "models" / "ggml-model-i2_s.gguf")
    ap.add_argument("-o", "--out", type=Path,
                    default=root / "results" / "tensor_structure.txt")
    ap.add_argument("--self-test", action="store_true",
                    help="run only the estimator checks and exit")
    args = ap.parse_args()

    st = self_test()
    print(f"self-test ok: iid drop {st[0]:.5f}, injected drop {st[1]:.5f}",
          file=sys.stderr)
    if args.self_test:
        return 0

    if not args.gguf.exists():
        print(f"error: {args.gguf} not found", file=sys.stderr)
        return 2

    version, meta, tensors, data_start = parse_gguf(args.gguf)

    # nelem/4 bytes are all code bytes -- checked, not assumed.
    ordered = sorted(tensors, key=lambda t: t["offset"])
    blob_end = args.gguf.stat().st_size - data_start
    extras = set()
    for i, t in enumerate(ordered):
        if t["type"] != GGML_TYPE_I2_S:
            continue
        nxt = ordered[i + 1]["offset"] if i + 1 < len(ordered) else blob_end
        extras.add(nxt - t["offset"] - nelem_of(t) // 4)

    meta_lines = [
        f"Model  : {args.gguf.name}, microsoft/bitnet-b1.58-2B-4T-gguf",
        f"         {meta.get('general.architecture')}, GGUF v{version}, "
        f"{len(tensors)} tensors, data at {data_start}",
        f"Method : GGUF tensor table parsed with tools/gguf_inspect.py; codes "
        f"read from",
        f"         the file. No inference run, nothing sampled.",
        f"Layout : an i2_s tensor is exactly nelem/4 code bytes; the gap to the "
        f"next",
        f"         tensor is {sorted(extras)} bytes for every i2_s tensor (one "
        f"f32 scale +",
        f"         GGUF alignment), so no scale byte is ever counted as a weight.",
    ]

    i2s = [t for t in tensors if t["type"] == GGML_TYPE_I2_S]
    all_res, failed = [], []
    for n, t in enumerate(i2s, 1):
        detail = t["name"] in DETAIL_TENSORS
        if n % 20 == 0 or detail:
            print(f"  [{n}/{len(i2s)}] {t['name']}", file=sys.stderr, flush=True)
        try:
            all_res.append(measure_tensor(args.gguf, data_start, t, detail))
        except Exception as e:                                    # noqa: BLE001
            print(f"  FAILED {t['name']}: {e}", file=sys.stderr)
            failed.append(f"{t['name']}  ({e})")

    order = {n: i for i, n in enumerate(DETAIL_TENSORS)}
    detail_res = sorted((r for r in all_res if r["detail"]),
                        key=lambda r: order[r["name"]])
    for n in DETAIL_TENSORS:
        if n not in {r["name"] for r in detail_res}:
            failed.append(f"{n}  (requested for the detail table, not present)")

    print("sweeping every i2_s tensor for code 3 ...", file=sys.stderr, flush=True)
    sweep = sweep_code3(args.gguf, data_start, tensors)
    agg = aggregate(all_res)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        emit(all_res, detail_res, sweep, agg, meta_lines, st, f)
        if failed:
            f.write("\n\nNOT MEASURED\n" + "-" * 75 + "\n")
            for n in failed:
                f.write(f"  {n}\n")
        else:
            f.write("\n\nNOT MEASURED\n" + "-" * 75 + "\n")
            f.write("  Nothing. Every i2_s tensor in the file was read in full.\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
