#!/usr/bin/env python3
"""
gguf_to_tq1_0.py -- rewrite a BitNet i2_s GGUF into llama.cpp's OWN sub-two-bit
ternary type, TQ1_0, so that the two can be compared on the same machine
through the same binary.

  convert : python3 tools/gguf_to_tq1_0.py models/ggml-model-i2_s.gguf \
                                           models/ggml-model-tq1_0.gguf
  plan    : python3 tools/gguf_to_tq1_0.py models/ggml-model-i2_s.gguf --dry-run
  verify  : python3 tools/gguf_to_tq1_0.py models/ggml-model-i2_s.gguf --verify

WHY THIS EXISTS
---------------
This project built a 1.600-bit ternary packing (t5b, type 43) and reported it
against i2_s at 2.0 bits. A reviewer pointed out that llama.cpp already carries
a sub-two-bit ternary type of its own -- TQ1_0, 1.6875 bits/weight, merged
upstream in 2024 -- and that it is the nearest prior art. It sits in the very
tree this project vendored and compiled against; it was simply not noticed.

The honest comparison is TQ1_0 against t5b on this machine, through the same
binary, on the same weights. This tool produces the TQ1_0 file that comparison
needs.

WHY NO REQUANTISATION IS INVOLVED
---------------------------------
llama-quantize would want the bf16 checkpoint, ~5 GB that is not here and does
not need to be. The ternary weights already exist in the i2_s file, and TQ1_0
stores the SAME ternary values under the same radix-3 idea -- five trits to a
byte, 3^5 = 243 < 256 -- differing only in the block layout and in where the
scale lives. Writing the existing trits straight into TQ1_0 blocks is exact and
costs one pass over the file. Nothing is rounded except the scale (see below).

THE TWO LAYOUTS, READ FROM THE VENDORED TREE (NOT FROM MEMORY)
--------------------------------------------------------------
i2_s, from src_modifications/ggml_i2s_ternary.h: a 32-byte block over 128
weights; for byte j, bits 6-7 hold weight 0+j, bits 4-5 hold 32+j, bits 2-3
hold 64+j, bits 0-1 hold 96+j.  Code c maps to weight c-1.  The per-tensor f32
scale lives in a 32-byte tail after the codes (ggml.c:1316 sizes an i2_s tensor
as nbytes/4 + 32; ggml-cpu.c:1505 reads the scale at ne00*ne01/4).

TQ1_0, from ggml-common.h:266 --

    typedef struct {
        uint8_t qs[(QK_K - 4*QK_K/64)/5];  // 5 elements per byte
        uint8_t qh[QK_K/64];               // 4 elements per byte
        ggml_half d;
    } block_tq1_0;

QK_K = 256, so 48 + 4 + 2 = 54 bytes per 256 weights = 1.6875 bits/weight, and
GGML_TYPE_TQ1_0 = 34 (ggml.h:424).  There is no tail: the scale is per block.

THE PACKING IS DERIVED FROM THE DECODER, NOT FROM THE ENCODER
-------------------------------------------------------------
dequantize_row_tq1_0 (ggml-quants.c:2356) is what will read this file, so it is
the authority.  It emits, per block, in this order:

    qs[0..31] , five trits each, output index n*32 + m        -> weights 0..159
    qs[32..47], five trits each, output index 160 + n*16 + m  -> weights 160..239
    qh[0..3]  , four trits each, output index 240 + n*4 + j   -> weights 240..255

and extracts trit n of a byte b as

    uint8_t q = b * pow3[n];          /* wraps mod 256 -- deliberate */
    int16_t xi = ((uint16_t) q * 3) >> 8;

which is a fixed-point base-3 shift: the byte is ceil(v * 256 / 243) for the
5-trit word v, multiplying by 3^n drops the leading n trits off the top, and
the >> 8 reads the new leading trit.  The qh tail is the same scheme with the
five-trit word's LAST digit forced to zero (the encoder's `q *= 3` at
ggml-quants.c:2301), which is why its four trits are read with pow3[0..3].

_encode_5 / _encode_4 / _decode_trit below transcribe that, and the tool checks
the transcription exhaustively over all 243 five-trit words and all 81 tail
words before it writes a byte -- see check_codec().

THE ONE THING THAT IS NOT EXACT: THE SCALE
------------------------------------------
i2_s carries one f32 scale per tensor.  TQ1_0 carries an f16 scale per block.
Since the i2_s scale is constant across a tensor, putting it in every block
reproduces the same weights -- but f16 has 11 significant bits, so the scale
itself is rounded.  The tool measures that rounding per tensor and prints the
worst case rather than leaving it implied; over the shipped model the scales
are 210 floats in [0.744, 4.596] and the relative error is bounded by 2^-11.
This is a property of TQ1_0's design, not a defect of the conversion, and it is
the reason the two files' logits are close but not bit-identical.

REUSE
-----
The GGUF reader is tools/gguf_inspect.py's parse_gguf and the header-patch
helpers are tools/gguf_to_t5b.py's -- imported, not copied.  This file adds the
TQ1_0 geometry and the trit packing and nothing else.
"""

import argparse
import collections
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_inspect import parse_gguf                                  # noqa: E402
from gguf_to_t5b import (ConversionError, GGML_TYPE_I2_S,            # noqa: E402
                         GGUF_DEFAULT_ALIGNMENT, align_up,
                         check_size_model, i2s_unpack_codes,
                         locate_tensor_fields, nelem, plan_layout,
                         t5b_nbytes, tensor_nbytes)

# --------------------------------------------------------------------------
# TQ1_0, from the vendored tree
# --------------------------------------------------------------------------

# ggml.h:424. Not a free slot this project chose -- an upstream type that has
# been in llama.cpp since 2024, with .from_float, .to_float and a
# ggml_vec_dot_tq1_0_q8_K already wired into ggml-cpu.c:401.
GGML_TYPE_TQ1_0 = 34

QK_K = 256                       # ggml-common.h
TQ1_QS = (QK_K - 4 * QK_K // 64) // 5      # 48 bytes, five trits each
TQ1_QH = QK_K // 64                        # 4 bytes, four trits each
TQ1_BLOCK_BYTES = TQ1_QS + TQ1_QH + 2      # + ggml_half d = 54

POW3 = np.array([1, 3, 9, 27, 81, 243], dtype=np.uint16)

# Weights of the five trits inside a byte, most significant first: the encoder
# at ggml-quants.c:2265 does `q *= 3; q += xi` with n ascending, so the trit
# read back at n = 0 is the one worth 81.
W5 = np.array([81, 27, 9, 3, 1], dtype=np.uint32)
# The tail packs four trits and then shifts left one place (`q *= 3`), so the
# same weights apply with the units place left empty.
W4 = np.array([81, 27, 9, 3], dtype=np.uint32)


# --------------------------------------------------------------------------
# The codec, transcribed from ggml-quants.c and checked against itself
# --------------------------------------------------------------------------

def _encode_word(v):
    """ceil(v * 256 / 243), the encoder's `q = ((uint16_t)q*256 + 242)/243`."""
    return (v * 256 + 242) // 243


def _decode_trit(byte, n):
    """dequantize_row_tq1_0's extraction of trit n. The & 0xFF is the C
    `uint8_t q = ... * pow3[n]` truncation and is load-bearing."""
    q = (byte * int(POW3[n])) & 0xFF
    return (q * 3) >> 8


def check_codec():
    """Every five-trit word and every four-trit tail word, encoded and read
    back through the decoder's own arithmetic.

    243 + 81 words is the WHOLE domain, so this is not a sample. It exists
    because the packing is derived from a C function by hand, and a hand
    transcription that is wrong in one digit position produces a file that
    still loads and generates fluent nonsense.
    """
    n5 = n4 = 0
    for v in range(243):
        ds = [(v // 81) % 3, (v // 27) % 3, (v // 9) % 3, (v // 3) % 3, v % 3]
        b = _encode_word(v)
        if b > 255:
            raise ConversionError(f"five-trit word {v} encodes to {b} > 255")
        if [_decode_trit(b, n) for n in range(5)] != ds:
            raise ConversionError(f"five-trit word {v} does not read back")
        n5 += 1
    for v in range(81):
        ds = [(v // 27) % 3, (v // 9) % 3, (v // 3) % 3, v % 3]
        b = _encode_word(v * 3)
        if b > 255:
            raise ConversionError(f"tail word {v} encodes to {b} > 255")
        if [_decode_trit(b, n) for n in range(4)] != ds:
            raise ConversionError(f"tail word {v} does not read back")
        n4 += 1
    return n5, n4


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def tq1_row_size(ne0):
    """ggml_row_size(GGML_TYPE_TQ1_0, ne0) = type_size * ne0/blck_size."""
    if ne0 % QK_K:
        raise ConversionError(f"ne0 = {ne0} is not a multiple of QK_K = {QK_K}; "
                              f"TQ1_0 rows would not start on a block boundary")
    return ne0 // QK_K * TQ1_BLOCK_BYTES


def tq1_nbytes(dims):
    """ggml_nbytes for a contiguous 2-D TQ1_0 tensor. No 32-byte tail: TQ1_0
    puts the scale in every block, so the i2_s tail has nowhere to go and
    nothing left to carry."""
    return tq1_row_size(dims[0]) * dims[1]


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------

def tq1_pack_codes(codes, d_f16_bytes):
    """Codes in {0,1,2}, shape (n,) with n % 256 == 0 -> TQ1_0 block bytes.

    Vectorised: 2.08 billion weights do not survive a Python loop. The three
    regions match the decoder's three loops exactly; see the module docstring.
    """
    n = codes.size
    if n % QK_K:
        raise ConversionError(f"{n} codes is not a multiple of {QK_K}")
    nb = n // QK_K
    c = codes.reshape(nb, QK_K).astype(np.uint32)

    out = np.empty((nb, TQ1_BLOCK_BYTES), dtype=np.uint8)

    # qs[0..31]: decoder index n*32 + m, five trits, m the byte, n the digit.
    v = (c[:, 0:160].reshape(nb, 5, 32) * W5.reshape(1, 5, 1)).sum(axis=1)
    out[:, 0:32] = ((v * 256 + 242) // 243).astype(np.uint8)

    # qs[32..47]: decoder index 160 + n*16 + m.
    v = (c[:, 160:240].reshape(nb, 5, 16) * W5.reshape(1, 5, 1)).sum(axis=1)
    out[:, 32:48] = ((v * 256 + 242) // 243).astype(np.uint8)

    # qh[0..3]: decoder index 240 + n*4 + j, four trits in the top four places.
    v = (c[:, 240:256].reshape(nb, 4, 4) * W4.reshape(1, 4, 1)).sum(axis=1)
    out[:, 48:52] = ((v * 256 + 242) // 243).astype(np.uint8)

    out[:, 52] = d_f16_bytes[0]
    out[:, 53] = d_f16_bytes[1]
    return out.reshape(-1)


def tq1_unpack_codes(packed, n):
    """TQ1_0 block bytes -> codes in {0,1,2}, shape (n,).

    A NumPy transcription of dequantize_row_tq1_0 (ggml-quants.c:2356), used to
    check the packing against the function that will actually read the file.
    It deliberately does NOT reuse tq1_pack_codes' arithmetic -- it goes
    through `(byte * pow3[n]) & 0xFF` and `(q*3) >> 8`, the decoder's own path,
    so a shared misunderstanding cannot cancel out.
    """
    nb = n // QK_K
    b = packed.reshape(nb, TQ1_BLOCK_BYTES)
    out = np.empty((nb, QK_K), dtype=np.uint8)

    qs = b[:, 0:48].astype(np.uint16)
    for dig in range(5):
        q = (qs[:, 0:32] * POW3[dig]) & 0xFF
        out[:, dig * 32:(dig + 1) * 32] = ((q * 3) >> 8).astype(np.uint8)
    for dig in range(5):
        q = (qs[:, 32:48] * POW3[dig]) & 0xFF
        out[:, 160 + dig * 16:160 + (dig + 1) * 16] = ((q * 3) >> 8).astype(np.uint8)

    qh = b[:, 48:52].astype(np.uint16)
    for dig in range(4):
        q = (qh * POW3[dig]) & 0xFF
        out[:, 240 + dig * 4:240 + (dig + 1) * 4] = ((q * 3) >> 8).astype(np.uint8)

    return out.reshape(-1)


def block_scales(packed, n):
    """The f16 `d` of every block, as float32."""
    nb = n // QK_K
    b = packed.reshape(nb, TQ1_BLOCK_BYTES)
    return b[:, 52:54].copy().view(np.float16).reshape(-1).astype(np.float32)


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------

def convert(src, dst, dry_run=False, verbose_plan=False, out=sys.stdout,
            check_every=3, check_positions=200, seed=20260906):
    version, meta, tensors, data_start = parse_gguf(src)
    file_size = src.stat().st_size

    n5, n4 = check_codec()
    print(f"codec check   : {n5} five-trit words and {n4} tail words encoded "
          f"and read back through dequantize_row_tq1_0's arithmetic, "
          f"0 mismatches", file=out)

    alignment = meta.get("general.alignment")
    align_src = "general.alignment in the KV block"
    if alignment is None:
        alignment = GGUF_DEFAULT_ALIGNMENT
        align_src = ("absent from the KV block -- the GGUF spec default of 32 "
                     "applies")
    if alignment <= 0 or (alignment & (alignment - 1)):
        raise ConversionError(f"alignment {alignment} is not a power of two")

    i2s = [t for t in tensors if t["type"] == GGML_TYPE_I2_S]
    if not i2s:
        raise ConversionError(f"{src}: no i2_s tensors to convert")

    for t in i2s:
        if len(t["dims"]) != 2:
            raise ConversionError(
                f"{t['name']}: {len(t['dims'])} dimensions, expected 2.")
        if t["dims"][0] % QK_K:
            raise ConversionError(
                f"{t['name']}: ne0 = {t['dims'][0]} is not a multiple of "
                f"{QK_K}. TQ1_0 blocks would straddle rows and llama.cpp would "
                f"reject the tensor at load; refusing to write it.")

    total_before = check_size_model(tensors, alignment, data_start, file_size)

    def new_size(t):
        return tq1_nbytes(t["dims"]) if t["type"] == GGML_TYPE_I2_S \
            else tensor_nbytes(t)

    new_offsets, total_after = plan_layout(tensors, new_size, alignment)

    # ---- the arithmetic ---------------------------------------------------
    shapes = collections.Counter(tuple(t["dims"]) for t in i2s)
    n_weights = sum(nelem(t["dims"]) for t in i2s)
    print(f"source        : {src}  ({file_size:,} bytes)", file=out)
    print(f"GGUF v{version}, {len(tensors)} tensors, data starts at "
          f"{data_start:,}", file=out)
    print(f"alignment     : {alignment}  ({align_src})", file=out)
    print(f"i2_s tensors  : {len(i2s)} of {len(tensors)}   "
          f"{n_weights:,} ternary weights", file=out)
    print(f"target type   : {GGML_TYPE_TQ1_0} (GGML_TYPE_TQ1_0), "
          f"{TQ1_BLOCK_BYTES} bytes per {QK_K} weights = "
          f"{TQ1_BLOCK_BYTES * 8 / QK_K:.4f} bits/weight", file=out)

    print("\n  K       rows/shape   i2_s bytes/row  tq1_0 bytes/row  "
          "t5b bytes/row", file=out)
    for dims, cnt in sorted(shapes.items()):
        k = dims[0]
        print(f"  {k:<7} x{cnt:<10} {k // 4:>14,} {tq1_row_size(k):>16,} "
              f"{(t5b_nbytes([k, 1]) - 32):>14,}", file=out)

    b_i2s = sum(nelem(t["dims"]) // 4 for t in i2s)
    b_tq1 = sum(tq1_nbytes(t["dims"]) for t in i2s)
    b_t5b = sum(t5b_nbytes(t["dims"]) - 32 for t in i2s)
    print(f"\n  ternary payload  i2_s {b_i2s:>13,} B  "
          f"{b_i2s * 8 / n_weights:.5f} bits/weight", file=out)
    print(f"                   tq1_0 {b_tq1:>12,} B  "
          f"{b_tq1 * 8 / n_weights:.5f} bits/weight", file=out)
    print(f"                   t5b  {b_t5b:>13,} B  "
          f"{b_t5b * 8 / n_weights:.5f} bits/weight", file=out)
    print(f"  the i2_s and t5b files also carry a 32-byte tail per tensor "
          f"({len(i2s) * 32:,} B); TQ1_0 has none.", file=out)
    print(f"file  : {file_size:,} -> {data_start + total_after:,} bytes  "
          f"({(data_start + total_after) / file_size:.5f}, "
          f"{(file_size - (data_start + total_after)) / 1e6:.1f} MB saved)",
          file=out)

    if verbose_plan:
        print("\nPER-TENSOR PLAN (file order)", file=out)
        for t in sorted(tensors, key=lambda t: t["offset"]):
            print(f"  {t['name']:<34} {str(t['dims']):<16} "
                  f"type {t['type']:>2} -> "
                  f"{GGML_TYPE_TQ1_0 if t['type'] == GGML_TYPE_I2_S else t['type']:>2}   "
                  f"off {t['offset']:>12,} -> {new_offsets[t['name']]:>12,}   "
                  f"size {tensor_nbytes(t):>12,} -> {new_size(t):>12,}",
                  file=out)

    if dry_run:
        print("\n--dry-run: nothing written.", file=out)
        return None

    # ---- header: copy verbatim, patch two fixed-width fields per tensor ----
    fields, table_end = locate_tensor_fields(src, tensors)
    with open(src, "rb") as f:
        header = bytearray(f.read(data_start))
    for fld in fields:
        if fld["type_pos"] + 4 > data_start or fld["offset_pos"] + 8 > data_start:
            raise ConversionError(f"{fld['name']}: tensor entry runs past the "
                                  f"data start; refusing to patch")
        if fld["type"] == GGML_TYPE_I2_S:
            struct.pack_into("<I", header, fld["type_pos"], GGML_TYPE_TQ1_0)
        struct.pack_into("<Q", header, fld["offset_pos"],
                         new_offsets[fld["name"]])

    # ---- data -------------------------------------------------------------
    rng = np.random.default_rng(seed)
    ordered = sorted(tensors, key=lambda t: t["offset"])
    t_start = time.time()
    checked = []            # (name, positions, sampled bad, ne, exhaustive bad)
    worst_scale = None      # (name, f32, f16, rel_err)
    n_i2s_seen = 0
    n_check = max(0, min(check_every, len(i2s)))
    check_at = {round(i * (len(i2s) - 1) / max(1, n_check - 1)) if n_check > 1
                else 0 for i in range(n_check)}

    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(header)
        for idx, t in enumerate(ordered):
            want = data_start + new_offsets[t["name"]]
            here = fout.tell()
            if here > want:
                raise ConversionError(f"{t['name']}: already past its offset "
                                      f"({here} > {want})")
            fout.write(b"\0" * (want - here))

            fin.seek(data_start + t["offset"])
            if t["type"] != GGML_TYPE_I2_S:
                remaining = tensor_nbytes(t)
                while remaining:
                    chunk = fin.read(min(remaining, 32 * 1024 * 1024))
                    if len(chunk) == 0:
                        raise ConversionError(f"{t['name']}: source truncated")
                    fout.write(chunk)
                    remaining -= len(chunk)
                continue

            k, rows = t["dims"][0], t["dims"][1]
            ne = k * rows
            raw = np.frombuffer(fin.read(ne // 4), dtype=np.uint8)
            tail = fin.read(32)
            if raw.size != ne // 4 or len(tail) != 32:
                raise ConversionError(f"{t['name']}: source truncated")

            codes = i2s_unpack_codes(raw, ne)
            n3 = int(np.count_nonzero(codes == 3))
            if n3:
                first = int(np.argmax(codes == 3))
                raise ConversionError(
                    f"{t['name']}: code 3 occurs {n3} times, first at weight "
                    f"{first}. TQ1_0 has three states per trit and no room "
                    f"for a fourth. Refusing.")

            # The scale: the first four bytes of the i2_s tail
            # (ggml-cpu.c:1505). It becomes every block's f16 d.
            scale32 = struct.unpack("<f", tail[:4])[0]
            if not math.isfinite(scale32) or scale32 <= 0.0:
                raise ConversionError(
                    f"{t['name']}: i2_s scale is {scale32!r}, which is not a "
                    f"finite positive float. Writing it into f16 would lose "
                    f"the tensor silently.")
            d16 = np.float16(scale32)
            if not np.isfinite(d16) or float(d16) == 0.0:
                raise ConversionError(
                    f"{t['name']}: scale {scale32!r} does not survive f16 "
                    f"(-> {float(d16)!r}); TQ1_0 cannot hold this tensor.")
            rel = abs(float(d16) - scale32) / scale32
            if worst_scale is None or rel > worst_scale[3]:
                worst_scale = (t["name"], scale32, float(d16), rel)
            d_bytes = np.array([d16], dtype=np.float16).view(np.uint8)

            packed = tq1_pack_codes(codes, d_bytes)
            if packed.size != tq1_nbytes(t["dims"]):
                raise ConversionError(f"{t['name']}: packed {packed.size} "
                                      f"bytes, expected {tq1_nbytes(t['dims'])}")

            # ---- the check that matters: the DECODER reads back the trits --
            # Spread the checked tensors over the file rather than taking the
            # first few: a packing bug that depends on K would hide behind
            # three consecutive tensors of the same shape.
            if n_i2s_seen in check_at:
                back = tq1_unpack_codes(packed, ne)
                pos = rng.choice(ne, size=min(check_positions, ne),
                                 replace=False)
                bad = int(np.count_nonzero(back[pos] != codes[pos]))
                full = int(np.count_nonzero(back != codes))
                checked.append((t["name"], len(pos), bad, ne, full))
            n_i2s_seen += 1

            fout.write(packed.tobytes())

            if (idx + 1) % 25 == 0:
                print(f"  ... {idx + 1}/{len(ordered)} tensors, "
                      f"{fout.tell() / 1e9:.2f} GB, "
                      f"{time.time() - t_start:.0f}s", file=out, flush=True)

        end = fout.tell()

    if end != data_start + total_after:
        raise ConversionError(f"wrote {end} bytes, planned "
                              f"{data_start + total_after}")

    print(f"\nwrote {dst}  ({end:,} bytes, {time.time() - t_start:.0f}s)",
          file=out)

    # ---- what the decoder said -------------------------------------------
    print("\nENCODER CHECKED AGAINST THE DECODER "
          "(dequantize_row_tq1_0, transcribed)", file=out)
    tot_pos = tot_bad = tot_all = tot_allbad = 0
    for name, npos, bad, ne, full in checked:
        print(f"  {name:<34} {npos} random positions: {bad} mismatches   "
              f"(all {ne:,} weights: {full} mismatches)", file=out)
        tot_pos += npos
        tot_bad += bad
        tot_all += ne
        tot_allbad += full
    print(f"  TOTAL {len(checked)} tensors, {tot_pos} sampled positions, "
          f"{tot_bad} mismatches; {tot_all:,} weights checked exhaustively, "
          f"{tot_allbad} mismatches", file=out)
    if tot_bad or tot_allbad:
        raise ConversionError("the decoder does not read back what was "
                              "written. The file is wrong; not shipping it.")

    name, s32, s16, rel = worst_scale
    print(f"\nSCALE, f32 -> f16 (the only lossy step)", file=out)
    print(f"  worst tensor {name}: {s32!r} -> {s16!r}, "
          f"relative error {rel:.3e}", file=out)
    print(f"  every block of a tensor carries the same d, so the ternary "
          f"weights are exact and only the per-tensor scale moves.", file=out)

    # ---- reread with the same parser --------------------------------------
    v2, meta2, tensors2, ds2 = parse_gguf(dst)
    if ds2 != data_start:
        raise ConversionError(f"data start moved: {data_start} -> {ds2}")
    n_tq1 = sum(1 for t in tensors2 if t["type"] == GGML_TYPE_TQ1_0)
    if n_tq1 != len(i2s):
        raise ConversionError(f"reread: {n_tq1} tq1_0 tensors, expected "
                              f"{len(i2s)}")
    if any(t["type"] == GGML_TYPE_I2_S for t in tensors2):
        raise ConversionError("reread: i2_s tensors remain")
    by_name = {t["name"]: t for t in tensors2}
    for t in tensors:
        r = by_name[t["name"]]
        if r["dims"] != t["dims"]:
            raise ConversionError(f"reread: {t['name']} dims changed")
        if r["offset"] != new_offsets[t["name"]]:
            raise ConversionError(f"reread: {t['name']} offset is "
                                  f"{r['offset']}, planned "
                                  f"{new_offsets[t['name']]}")
    print(f"\nreread: {n_tq1} tensors are type {GGML_TYPE_TQ1_0}, no i2_s "
          f"left, every dim and offset as planned.", file=out)
    return dst


# --------------------------------------------------------------------------
# Standalone verification of a written file
# --------------------------------------------------------------------------

def verify(src, dst, n_tensors, n_positions, seed, out=sys.stdout):
    """Re-open BOTH files and check the TQ1_0 trits against the i2_s trits,
    without going through anything the conversion computed."""
    _, _, ts_src, ds_src = parse_gguf(src)
    _, _, ts_dst, ds_dst = parse_gguf(dst)
    by_dst = {t["name"]: t for t in ts_dst}

    n5, n4 = check_codec()
    print(f"codec: {n5} + {n4} words, 0 mismatches", file=out)

    i2s = [t for t in ts_src if t["type"] == GGML_TYPE_I2_S]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(i2s), size=min(n_tensors, len(i2s)), replace=False)

    tot_pos = tot_bad = tot_all = tot_allbad = 0
    for i in sorted(pick):
        t = i2s[i]
        d = by_dst[t["name"]]
        if d["type"] != GGML_TYPE_TQ1_0:
            raise ConversionError(f"{t['name']} is type {d['type']} in {dst}")
        k, rows = t["dims"][0], t["dims"][1]
        ne = k * rows
        with open(src, "rb") as f:
            f.seek(ds_src + t["offset"])
            raw = np.frombuffer(f.read(ne // 4), dtype=np.uint8)
            tail = f.read(32)
        with open(dst, "rb") as f:
            f.seek(ds_dst + d["offset"])
            packed = np.frombuffer(f.read(tq1_nbytes(t["dims"])), dtype=np.uint8)

        codes = i2s_unpack_codes(raw, ne)
        back = tq1_unpack_codes(packed, ne)
        pos = rng.choice(ne, size=min(n_positions, ne), replace=False)
        bad = int(np.count_nonzero(back[pos] != codes[pos]))
        full = int(np.count_nonzero(back != codes))

        s32 = struct.unpack("<f", tail[:4])[0]
        ds = block_scales(packed, ne)
        want16 = float(np.float16(s32))
        n_wrong_d = int(np.count_nonzero(ds != want16))

        print(f"  {t['name']:<34} {len(pos)} sampled: {bad} bad; "
              f"all {ne:,}: {full} bad; scale {s32:.6f} -> f16 {want16:.6f}, "
              f"{n_wrong_d} of {ne // QK_K:,} blocks carry a different d",
              file=out)
        if n_wrong_d:
            raise ConversionError(f"{t['name']}: block scales are not uniform")
        tot_pos += len(pos)
        tot_bad += bad
        tot_all += ne
        tot_allbad += full

    print(f"TOTAL: {len(pick)} tensors, {tot_pos} sampled positions, "
          f"{tot_bad} mismatches; {tot_all:,} weights compared in full, "
          f"{tot_allbad} mismatches", file=out)
    return 1 if (tot_bad or tot_allbad) else 0


def main():
    ap = argparse.ArgumentParser(
        description="Rewrite a BitNet i2_s GGUF as llama.cpp's own TQ1_0.")
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path, nargs="?")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the arithmetic, write nothing")
    ap.add_argument("--plan", action="store_true",
                    help="with --dry-run, list every tensor")
    ap.add_argument("--verify", action="store_true",
                    help="check an already-written file against the source")
    ap.add_argument("--tensors", type=int, default=3,
                    help="how many tensors to check (default 3)")
    ap.add_argument("--positions", type=int, default=200,
                    help="random positions per tensor (default 200)")
    ap.add_argument("--seed", type=int, default=20260906)
    a = ap.parse_args()

    if a.verify:
        if a.dst is None:
            ap.error("--verify needs both files")
        return verify(a.src, a.dst, a.tensors, a.positions, a.seed)

    if not a.dry_run and a.dst is None:
        ap.error("give an output path, or pass --dry-run")
    try:
        convert(a.src, a.dst, dry_run=a.dry_run, verbose_plan=a.plan,
                check_every=a.tensors, check_positions=a.positions,
                seed=a.seed)
    except ConversionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
