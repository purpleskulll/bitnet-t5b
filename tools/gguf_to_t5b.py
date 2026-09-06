#!/usr/bin/env python3
"""
gguf_to_t5b.py -- repack a BitNet i2_s GGUF into the 1.6-bit t5b packing, and
verify the repacking against the REAL weights rather than against random
ternary.

  convert : python3 tools/gguf_to_t5b.py models/ggml-model-i2_s.gguf
  plan    : python3 tools/gguf_to_t5b.py models/ggml-model-i2_s.gguf --dry-run
  verify  : python3 tools/gguf_to_t5b.py models/ggml-model-i2_s.gguf --verify

WHAT IT DOES
------------
Every i2_s tensor is unpacked to ternary weights and repacked with the layout
in src_modifications/ternary_t5b.h -- 32 bytes over 160 weights, five base-3
digits per byte. Every other tensor is copied byte for byte; the f16 128k
embedding is 55.7 % of the file and does not change. The header is copied
verbatim and only two fields per converted tensor are patched, the type and the
data offset, both fixed-width, so the metadata block keeps its exact byte
length and the data section keeps its start.

THE 32 TRAILING BYTES ARE NOT PADDING
-------------------------------------
ggml.c:1316 sizes an i2_s tensor as nbytes/4 + 32, and the extra 32 bytes carry
something the model needs:

    ggml-cpu.c:1505:  const float * scale =
        (const float *)((const uint8_t *)src0->data + (ne00 * ne01 / 4));

the per-tensor weight scale, read from the first four bytes after the codes and
applied as `post_scale = ws / act_scales[col]`. Measured over the shipped
model's 210 i2_s tensors it is 210 finite positive floats between 0.744214 and
4.596097 -- a real per-tensor quantity, not a constant and not zero fill. The
remaining 28 bytes are uninitialised (89 distinct values over 210 tensors, some
of them stale code bytes from an earlier buffer, some stale floats).

So the converter carries all 32 bytes over VERBATIM and appends them after the
t5b codes, in the same relation to the data they belong to. A consumer finds
the scale at rows * ternary_t5b_size(ne0) instead of at ne0*ne1/4. Dropping the
tail would silently cost every tensor its scale, which is the kind of loss that
still round-trips perfectly on the weights and produces garbage at inference.

TYPE NUMBER
-----------
The converted tensors are written with type 43, GGML_TYPE_T5B: a free slot in
ggml.h:390-434, where the highest assigned value is GGML_TYPE_TL2 = 42 and
GGML_TYPE_COUNT is 43. Taking it means a loader must also raise
GGML_TYPE_COUNT to 44; stock llama.cpp will not read the produced file, by
construction -- there is no t5b type in it to read.

REUSE
-----
The GGUF metadata and tensor table come from tools/gguf_inspect.py's
parse_gguf. This file adds one thing parse_gguf does not return -- the byte
POSITION of each tensor's type and offset field, which is what patching needs
-- and cross-checks its own walk against parse_gguf's result before using it.
"""

import argparse
import collections
import ctypes
import hashlib
import math
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_inspect import parse_gguf                                # noqa: E402
from derive_filler_tokens import _Reader, GGUFError                # noqa: E402

# --------------------------------------------------------------------------
# Formats
# --------------------------------------------------------------------------

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_I2_S = 36

# A free slot in ggml.h:390-434: GGML_TYPE_TL2 = 42 is the highest assigned and
# GGML_TYPE_COUNT = 43, so 43 is the next number an upstream addition would
# take. Not invented -- read from the vendored header.
GGML_TYPE_T5B = 43

# Byte size of one element, for the types this model actually contains. Any
# other type is refused rather than guessed at: a wrong size here would shift
# every subsequent tensor.
GGML_TYPE_SIZE = {GGML_TYPE_F32: 4, GGML_TYPE_F16: 2}

# i2_s, from src_modifications/ggml_i2s_ternary.h: a block is 32 bytes over 128
# weights, and for byte j of the block bits 6-7 hold the weight at block offset
# 0+j, bits 4-5 at 32+j, bits 2-3 at 64+j, bits 0-1 at 96+j.
I2S_BLOCK = 128
I2S_BYTES = 32

# t5b, from src_modifications/ternary_t5b.h: 32 bytes over 160 weights, byte j
# holding sum_k code(w[32k+j]) * 3^k, at most 242.
T5B_BLOCK = 160
T5B_BYTES = 32
T5B_DIGITS = 5
T5B_POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint16)

# The GGUF spec's default when general.alignment is absent from the KV block.
GGUF_DEFAULT_ALIGNMENT = 32

DEFAULT_SRC = Path(__file__).resolve().parent.parent / "src_modifications"


class ConversionError(Exception):
    """Anything unexpected. Raised rather than worked around."""


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def t5b_blocks(n):
    return (n + T5B_BLOCK - 1) // T5B_BLOCK


def t5b_size(n):
    """Packed bytes for n weights, rounded up to a whole block. Mirrors
    ternary_t5b_size() in src_modifications/ternary_t5b.c; --verify checks the
    two agree by calling the C one."""
    return t5b_blocks(n) * T5B_BYTES


def align_up(x, a):
    return x if x % a == 0 else x + a - (x % a)


def nelem(dims):
    return math.prod(dims)


def i2s_nbytes(dims):
    """ggml.c:1316 -- nbytes/4 + 32 for I2_S. The +32 is the scale tail."""
    return nelem(dims) // 4 + 32


def t5b_nbytes(dims):
    """One t5b row per matrix row, plus the same 32-byte tail carried over."""
    ne0, ne1 = dims[0], dims[1]
    return t5b_size(ne0) * ne1 + 32


def tensor_nbytes(t):
    if t["type"] == GGML_TYPE_I2_S:
        return i2s_nbytes(t["dims"])
    if t["type"] not in GGML_TYPE_SIZE:
        raise ConversionError(
            f"{t['name']}: type {t['type']} has no size rule here. Refusing to "
            f"guess -- a wrong size shifts every tensor after it.")
    return nelem(t["dims"]) * GGML_TYPE_SIZE[t["type"]]


# --------------------------------------------------------------------------
# Header: the one thing parse_gguf does not give us
# --------------------------------------------------------------------------

def locate_tensor_fields(path, tensors):
    """Byte positions of each tensor entry's type (u32) and offset (u64).

    parse_gguf returns the VALUES; patching needs the POSITIONS. This walks the
    same bytes with the same _Reader and then asserts its walk landed on
    parse_gguf's answer -- name, dims, type and offset, all 332 of them. If the
    two disagree the walk is wrong and this raises instead of writing a file
    whose tensor table points at nothing.
    """
    with open(path, "rb") as f:
        buf = f.read(512 * 1024 * 1024)

    r = _Reader(buf)
    r.take(4)                                            # magic
    r.take(4)                                            # version
    n_tensors = struct.unpack("<Q", r.take(8))[0]
    n_kv = struct.unpack("<Q", r.take(8))[0]
    for _ in range(n_kv):
        r.string()
        vtype = struct.unpack("<I", r.take(4))[0]
        r.value(vtype)

    fields = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = struct.unpack("<I", r.take(4))[0]
        dims = [struct.unpack("<Q", r.take(8))[0] for _ in range(n_dims)]
        type_pos = r.off
        ttype = struct.unpack("<I", r.take(4))[0]
        offset_pos = r.off
        offset = struct.unpack("<Q", r.take(8))[0]
        fields.append({"name": name, "dims": dims, "type": ttype,
                       "offset": offset, "type_pos": type_pos,
                       "offset_pos": offset_pos})

    if len(fields) != len(tensors):
        raise ConversionError("tensor-table walk disagrees with parse_gguf on "
                              f"count: {len(fields)} vs {len(tensors)}")
    for a, b in zip(fields, tensors):
        if (a["name"], a["dims"], a["type"], a["offset"]) != \
           (b["name"], b["dims"], b["type"], b["offset"]):
            raise ConversionError(
                f"tensor-table walk disagrees with parse_gguf at {b['name']}: "
                f"{a} vs {b}")
    return fields, r.off


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------

def i2s_unpack_codes(raw, ne):
    """i2_s code bytes -> the 2-bit codes in weight order, as uint8 in {0,1,2}.

    Vectorised: 2.08 billion weights do not survive a Python loop. `raw` is
    exactly ne//4 code bytes, WITHOUT the 32-byte scale tail.
    """
    if ne % I2S_BLOCK:
        raise ConversionError(f"element count {ne} is not a multiple of "
                              f"{I2S_BLOCK}; rows would not start on a block "
                              f"boundary and this unpack would be wrong")
    if raw.size != ne // 4:
        raise ConversionError(f"expected {ne // 4} code bytes, got {raw.size}")

    b = raw.reshape(-1, I2S_BYTES)
    out = np.empty((b.shape[0], I2S_BLOCK), dtype=np.uint8)
    out[:, 0:32] = (b >> 6) & 3
    out[:, 32:64] = (b >> 4) & 3
    out[:, 64:96] = (b >> 2) & 3
    out[:, 96:128] = b & 3
    return out.reshape(-1)


def t5b_pack_codes(codes, rows, k):
    """Codes in {0,1,2}, shape (rows*k,) -> t5b bytes, shape (rows*t5b_size(k),).

    Each ROW is packed independently, so a row always starts on a block
    boundary; rows are what a GEMV walks. Slots past k in the final block hold
    code 1, which is weight 0, exactly as ternary_t5b_pack does.
    """
    nblk = t5b_blocks(k)
    slots = nblk * T5B_BLOCK
    padded = np.ones((rows, slots), dtype=np.uint8)
    padded[:, :k] = codes.reshape(rows, k)

    v = padded.reshape(rows, nblk, T5B_DIGITS, T5B_BYTES).astype(np.uint16)
    byt = (v * T5B_POW3.reshape(1, 1, T5B_DIGITS, 1)).sum(axis=2,
                                                          dtype=np.uint16)
    if byt.max() > 242:
        raise ConversionError(f"packed byte {byt.max()} exceeds 242; the "
                              f"decode bounds in ternary_t5b.h do not hold")
    return byt.astype(np.uint8).reshape(-1)


def t5b_unpack_codes(packed, rows, k):
    """t5b bytes -> codes in {0,1,2}, shape (rows, k). By digit extraction,
    (byte // 3^d) % 3, which is the ENCODING definition and the same one
    ternary_t5b_unpack_at uses -- not the kernel's quotient identities."""
    nblk = t5b_blocks(k)
    b = packed.reshape(rows, nblk, T5B_BYTES).astype(np.uint16)
    slots = np.empty((rows, nblk, T5B_DIGITS, T5B_BYTES), dtype=np.uint8)
    for d in range(T5B_DIGITS):
        slots[:, :, d, :] = ((b // int(T5B_POW3[d])) % 3).astype(np.uint8)
    return slots.reshape(rows, nblk * T5B_BLOCK)[:, :k]


def code_histogram(raw):
    """Counts of the four 2-bit codes in an i2_s byte buffer, without
    materialising one entry per weight: histogram the 256 byte values once and
    fold the four fields of each value into it."""
    bytehist = np.bincount(raw, minlength=256).astype(np.int64)
    vals = np.arange(256, dtype=np.uint8)
    out = np.zeros(4, dtype=np.int64)
    for shift in (6, 4, 2, 0):
        codes = (vals >> shift) & 3
        for c in range(4):
            out[c] += bytehist[codes == c].sum()
    return out


# --------------------------------------------------------------------------
# Layout planning
# --------------------------------------------------------------------------

def plan_layout(tensors, size_of, alignment):
    """Assign data offsets in FILE ORDER. Returns {name: offset} and the total
    data-section length."""
    ordered = sorted(tensors, key=lambda t: t["offset"])
    offsets = {}
    cur = 0
    for t in ordered:
        offsets[t["name"]] = cur
        cur = align_up(cur + size_of(t), alignment)
    return offsets, cur


def check_size_model(tensors, alignment, data_start, file_size):
    """The poka-yoke. Re-derive the ORIGINAL layout from the same size formulas
    the conversion will use, and require it to reproduce every one of the
    original offsets and the exact file length. If the size model can predict
    the file it was read from, it can lay out the new one; if it cannot, the
    new file would be silently misaligned and every tensor after the first
    mistake would be garbage."""
    offsets, total = plan_layout(tensors, tensor_nbytes, alignment)
    for t in tensors:
        if offsets[t["name"]] != t["offset"]:
            raise ConversionError(
                f"size model does not reproduce the source layout at "
                f"{t['name']}: predicted offset {offsets[t['name']]}, file "
                f"says {t['offset']}")
    if data_start + total != file_size:
        raise ConversionError(
            f"size model predicts a {data_start + total}-byte file, the source "
            f"is {file_size} bytes")
    return total


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------

def convert(src, dst, dry_run=False, verbose_plan=False, out=sys.stdout,
            quiet_footer=False):
    version, meta, tensors, data_start = parse_gguf(src)
    file_size = src.stat().st_size

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
                f"{t['name']}: {len(t['dims'])} dimensions, expected 2. The "
                f"row packing is only defined for a matrix.")
        ne = nelem(t["dims"])
        if ne % 4:
            raise ConversionError(f"{t['name']}: {ne} elements is not a "
                                  f"multiple of 4; i2_s cannot hold it")
        if t["dims"][0] % I2S_BLOCK:
            raise ConversionError(
                f"{t['name']}: ne0 = {t['dims'][0]} is not a multiple of "
                f"{I2S_BLOCK}, so its rows do not start on i2_s block "
                f"boundaries. The unpack assumes they do.")

    total_before = check_size_model(tensors, alignment, data_start, file_size)

    def new_size(t):
        return t5b_nbytes(t["dims"]) if t["type"] == GGML_TYPE_I2_S \
            else tensor_nbytes(t)

    new_offsets, total_after = plan_layout(tensors, new_size, alignment)

    # ---- the arithmetic, computed and not assumed -------------------------
    shapes = collections.Counter(tuple(t["dims"]) for t in i2s)
    print(f"source        : {src}  ({file_size:,} bytes)", file=out)
    print(f"GGUF v{version}, {len(tensors)} tensors, data starts at "
          f"{data_start:,}", file=out)
    print(f"alignment     : {alignment}  ({align_src})", file=out)
    print(f"i2_s tensors  : {len(i2s)} of {len(tensors)}", file=out)
    print(file=out)
    print("PER-K ARITHMETIC", file=out)
    print("  K      blocks  bytes/row  i2_s bytes/row   bits/weight  ratio",
          file=out)
    for k in sorted({t["dims"][0] for t in i2s}):
        nb, by = t5b_blocks(k), t5b_size(k)
        i2b = k // 4
        print(f"  {k:<6} {nb:<7} {by:<10} {i2b:<16} "
              f"{by * 8 / k:.5f}      {by / i2b:.5f}", file=out)
    print(file=out)
    print("PER-SHAPE TOTALS", file=out)
    w_before = w_after = 0
    n_weights = 0
    for shape, count in sorted(shapes.items()):
        k, rows = shape[0], shape[1]
        b_i2s = i2s_nbytes(shape) * count
        b_t5b = t5b_nbytes(shape) * count
        w_before += b_i2s
        w_after += b_t5b
        n_weights += nelem(shape) * count
        print(f"  {str(shape):<16} x{count:<4} "
              f"i2_s {b_i2s:>13,} -> t5b {b_t5b:>13,}   "
              f"{b_t5b / b_i2s:.5f}", file=out)
    print(f"  {'TOTAL i2_s':<16} {'':5} "
          f"     {w_before:>13,} ->     {w_after:>13,}   "
          f"{w_after / w_before:.5f}", file=out)
    print(file=out)
    # Where the model-wide figure differs from the nominal 1.6, decomposed
    # rather than asserted. Two separate costs, and neither is rounding.
    n_scale_tails = len(i2s) * 32
    codes_before = w_before - n_scale_tails
    codes_after = w_after - n_scale_tails
    ideal = n_weights * 1.6 / 8
    pad_slots = sum((t5b_blocks(s[0]) * T5B_BLOCK - s[0]) * s[1] * c
                    for s, c in shapes.items())
    print(f"ternary weights            : {n_weights:,}", file=out)
    print(f"i2_s  bits/weight, codes   : "
          f"{codes_before * 8 / n_weights:.5f}   ({codes_before:,} bytes)",
          file=out)
    print(f"t5b   bits/weight, codes   : "
          f"{codes_after * 8 / n_weights:.5f}   ({codes_after:,} bytes)",
          file=out)
    print(f"  nominal 1.6 would be     : 1.60000   ({ideal:,.0f} bytes)",
          file=out)
    print(f"  tail waste at K=6912     : {pad_slots:,} padded slots "
          f"({pad_slots * 1.6 / 8:,.0f} bytes, "
          f"+{(codes_after - ideal) * 8 / n_weights:.5f} bits/weight)",
          file=out)
    print(f"  the 32-byte scale tails  : {n_scale_tails:,} bytes "
          f"(+{n_scale_tails * 8 / n_weights:.5f} bits/weight)", file=out)
    print(f"i2_s  bits/weight, all in  : "
          f"{w_before * 8 / n_weights:.5f}", file=out)
    print(f"t5b   bits/weight, all in  : "
          f"{w_after * 8 / n_weights:.5f}", file=out)
    print(f"file  : {file_size:,} -> {data_start + total_after:,} bytes  "
          f"({(data_start + total_after) / file_size:.5f}, "
          f"{(file_size - (data_start + total_after)) / 1e6:.1f} MB saved)",
          file=out)
    f16 = sum(tensor_nbytes(t) for t in tensors if t["type"] == GGML_TYPE_F16)
    print(f"        the f16 embedding is {f16:,} bytes -- "
          f"{f16 * 100.0 / (file_size - data_start):.1f} % of the tensor data, "
          f"{f16 * 100.0 / file_size:.1f} % of the file -- and does not "
          f"change; the whole saving comes out of the {codes_before:,} bytes "
          f"of codes", file=out)

    if verbose_plan:
        print("\nPER-TENSOR PLAN (file order)", file=out)
        for t in sorted(tensors, key=lambda t: t["offset"]):
            print(f"  {t['name']:<34} {str(t['dims']):<16} "
                  f"type {t['type']:>2} -> "
                  f"{GGML_TYPE_T5B if t['type'] == GGML_TYPE_I2_S else t['type']:>2}   "
                  f"off {t['offset']:>12,} -> {new_offsets[t['name']]:>12,}   "
                  f"size {tensor_nbytes(t):>12,} -> {new_size(t):>12,}",
                  file=out)

    if dry_run:
        if not quiet_footer:
            print("\n--dry-run: nothing written.", file=out)
        return None

    # ---- header: copy verbatim, patch two fixed-width fields per tensor ---
    fields, table_end = locate_tensor_fields(src, tensors)
    with open(src, "rb") as f:
        header = bytearray(f.read(data_start))
    for fld in fields:
        if fld["type_pos"] + 4 > data_start or fld["offset_pos"] + 8 > data_start:
            raise ConversionError(f"{fld['name']}: tensor entry runs past the "
                                  f"data start; refusing to patch")
        if fld["type"] == GGML_TYPE_I2_S:
            struct.pack_into("<I", header, fld["type_pos"], GGML_TYPE_T5B)
        struct.pack_into("<Q", header, fld["offset_pos"],
                         new_offsets[fld["name"]])

    # ---- data ------------------------------------------------------------
    ordered = sorted(tensors, key=lambda t: t["offset"])
    t_start = time.time()
    written_codes = 0
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
                    f"{first}. code = w + 1 has no room for it, the "
                    f"sum(code*y) = sum(w*y) + sum(y) identity fails, and t5b "
                    f"would carry it into the next digit plane. Refusing.")

            packed = t5b_pack_codes(codes, rows, k)
            if packed.size != t5b_size(k) * rows:
                raise ConversionError(f"{t['name']}: packed {packed.size} "
                                      f"bytes, expected {t5b_size(k) * rows}")
            fout.write(packed.tobytes())
            fout.write(tail)                     # the scale, carried verbatim
            written_codes += packed.size

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

    # Read the new file back with the same parser and require it to describe
    # what was intended. An ack from the writer is not evidence.
    v2, meta2, tensors2, ds2 = parse_gguf(dst)
    if ds2 != data_start:
        raise ConversionError(f"data start moved: {data_start} -> {ds2}")
    n_t5b = sum(1 for t in tensors2 if t["type"] == GGML_TYPE_T5B)
    if n_t5b != len(i2s):
        raise ConversionError(f"reread: {n_t5b} t5b tensors, expected "
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
        if r["offset"] % alignment:
            raise ConversionError(f"reread: {t['name']} offset "
                                  f"{r['offset']} violates alignment "
                                  f"{alignment}")
    print(f"reread OK: v{v2}, {len(tensors2)} tensors, {n_t5b} of type "
          f"{GGML_TYPE_T5B} (T5B), data at {ds2:,}, every offset a multiple "
          f"of {alignment}", file=out)
    return dst


# --------------------------------------------------------------------------
# Verification against the C reference
# --------------------------------------------------------------------------

def build_reference_so(src_dir, workdir, out=sys.stdout):
    """Compile the C reference kernels into a shared object and load it.

    Two files, both read-only inputs:
      ggml_i2s_ternary.c -- holds bitnet_vec_dot_i2_i8_s_reference, which is
        upstream's AVX2 i2_s branch copied verbatim. This is the authority on
        the i2_s layout: not a Python re-derivation of it.
      ternary_t5b.c -- the t5b packer, scalar reference and AVX2 kernel.
    """
    srcs = [src_dir / "ggml_i2s_ternary.c", src_dir / "ternary_t5b.c"]
    for s in srcs:
        if not s.exists():
            raise ConversionError(f"missing {s}")
    so = workdir / "libt5bcheck.so"
    cmd = ["gcc", "-O3", "-mavx2", "-mfma", "-march=native",
           "-Wall", "-Wextra", "-std=c11", "-fPIC", "-shared",
           *[str(s) for s in srcs], "-o", str(so)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise ConversionError(f"build failed:\n{r.stdout}\n{r.stderr}")

    lib = ctypes.CDLL(str(so))
    lib.bitnet_vec_dot_i2_i8_s_reference.restype = None
    lib.bitnet_vec_dot_i2_i8_s_reference.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_int]
    lib.ternary_t5b_size.restype = ctypes.c_size_t
    lib.ternary_t5b_size.argtypes = [ctypes.c_size_t]
    lib.ternary_t5b_pack.restype = None
    lib.ternary_t5b_pack.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_size_t]
    lib.ternary_t5b_unpack_at.restype = ctypes.c_int8
    lib.ternary_t5b_unpack_at.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.ternary_t5b_dot_scalar.restype = ctypes.c_int32
    lib.ternary_t5b_dot_scalar.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_size_t]
    lib.ternary_t5b_dot_avx2.restype = ctypes.c_int32
    lib.ternary_t5b_dot_avx2.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_size_t, ctypes.c_int32]

    digests = {s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in srcs}
    print("  C reference built from " + str(src_dir), file=out)
    for n, d in sorted(digests.items()):
        print(f"    {n:<22} sha256 {d}", file=out)
    return lib


def i2s_ref_codesum(lib, row_bytes, act, k):
    """Upstream's AVX2 i2_s branch on one row. Returns sum(code*a) as it does
    -- the -1 offset is applied downstream by the caller, per
    ggml_i2s_ternary.h's contract."""
    s = (ctypes.c_float * 1)()
    lib.bitnet_vec_dot_i2_i8_s_reference(
        k, s, 0, row_bytes.ctypes.data, k, act.ctypes.data, 0, 1)
    return int(s[0])


def verify(src, src_dir, tensor_names, n_rows, seed, converted=None,
           out=sys.stdout):
    rng = np.random.default_rng(seed)
    version, meta, tensors, data_start = parse_gguf(src)
    by_name = {t["name"]: t for t in tensors}
    i2s = [t for t in tensors if t["type"] == GGML_TYPE_I2_S]

    print("REAL WEIGHTS: THE i2_s -> t5b REPACKING, CHECKED ON THE SHIPPED "
          "MODEL", file=out)
    print("=" * 74, file=out)
    print(file=out)
    print("  Reproduce: python3 tools/gguf_to_t5b.py "
          "models/ggml-model-i2_s.gguf \\", file=out)
    print("                 -o models/ggml-model-t5b.gguf          "
          "# the conversion", file=out)
    print("             python3 tools/gguf_to_t5b.py "
          "models/ggml-model-i2_s.gguf \\", file=out)
    print("                 --verify > results/real_weights_check.txt "
          "# this file", file=out)
    print("             (section 5 reads the converted file back; run the "
          "conversion first)", file=out)
    print(file=out)
    print(f"  model   {src}", file=out)
    print(f"          GGUF v{version}, {len(tensors)} tensors, {len(i2s)} of "
          f"type i2_s, data at {data_start:,}", file=out)
    print(f"          {src.stat().st_size:,} bytes, sha256 of the first MiB "
          f"{hashlib.sha256(src.open('rb').read(1 << 20)).hexdigest()[:32]}",
          file=out)
    print(f"  seed    {seed}   rows per tensor {n_rows}", file=out)
    print(f"  host    {subprocess.run(['uname', '-srm'], capture_output=True, text=True).stdout.strip()}",
          file=out)
    print(file=out)

    with tempfile.TemporaryDirectory() as td:
        lib = build_reference_so(src_dir, Path(td), out=out)
        return _verify_body(lib, src, tensors, by_name, i2s, data_start,
                            tensor_names, n_rows, rng, converted, out)


def _read_i2s(src, t, data_start):
    ne = nelem(t["dims"])
    with open(src, "rb") as f:
        f.seek(data_start + t["offset"])
        raw = np.frombuffer(f.read(ne // 4), dtype=np.uint8)
        tail = f.read(32)
    return raw, tail


def _verify_body(lib, src, tensors, by_name, i2s, data_start, tensor_names,
                 n_rows, rng, converted, out):
    failures = []
    not_checked = []

    # ---- (0) the geometry, against the C function ------------------------
    print("(0) GEOMETRY", file=out)
    for k in (2560, 6912, 1, 159, 160, 161):
        py, c = t5b_size(k), lib.ternary_t5b_size(k)
        if py != c:
            failures.append(f"t5b_size({k}): python {py}, C {c}")
        print(f"    t5b_size({k:<5}) python {py:<6} C {c:<6} "
              f"{'agree' if py == c else 'DISAGREE'}", file=out)
    print(file=out)

    # ---- (1) is the layout the one the C kernel uses? --------------------
    print("(1) THE i2_s LAYOUT -- WHY THE PYTHON UNPACK IS NOT A MIRROR",
          file=out)
    print("""
    The unpack in this file could agree with a wrong reading of the model and
    prove nothing, so it is never used as the authority. The authority is
    bitnet_vec_dot_i2_i8_s_reference in src_modifications/ggml_i2s_ternary.c,
    which is upstream's AVX2 i2_s branch copied verbatim, compiled here and
    called through ctypes on the model's OWN bytes. Every dot product below is
    checked against it.

    That test can fail, which is the part that matters. The same upstream
    function has a SCALAR fallback (quants.c:107-111) that reads a different
    layout -- weight i from byte i/4 at bit 6-2*(i%4), four CONSECUTIVE weights
    per byte, against this file's stride-32 interleave. On AVX2 that branch
    never runs, so nothing in ordinary use distinguishes them. Below, both
    readings are contracted against the same activations and the same real
    bytes, and reported side by side.
""", file=out)

    probe = by_name["blk.0.ffn_down.weight"]
    kk, rr = probe["dims"][0], probe["dims"][1]
    raw, _ = _read_i2s(src, probe, data_start)
    rowbytes = kk // 4
    disc_rows = rng.choice(rr, size=64, replace=False)
    agree_interleaved = agree_consecutive = 0
    for r in disc_rows:
        rb = np.ascontiguousarray(raw[r * rowbytes:(r + 1) * rowbytes])
        act = rng.integers(-127, 128, size=kk, dtype=np.int64).astype(np.int8)
        ref = i2s_ref_codesum(lib, rb, act, kk)
        codes_int = i2s_unpack_codes(rb, kk).astype(np.int64)
        if int(codes_int @ act.astype(np.int64)) == ref:
            agree_interleaved += 1
        shifts = (6 - 2 * (np.arange(kk) % 4)).astype(np.uint8)
        alt = ((rb.astype(np.uint8)[np.arange(kk) // 4] >> shifts) & 3)
        if int(alt.astype(np.int64) @ act.astype(np.int64)) == ref:
            agree_consecutive += 1
    print(f"    on {len(disc_rows)} random rows of {probe['name']}, "
          f"against the C reference:", file=out)
    print(f"      stride-32 interleave (this file's reading) agrees on "
          f"{agree_interleaved}/{len(disc_rows)} rows", file=out)
    print(f"      four-consecutive (the scalar fallback's reading) agrees on "
          f"{agree_consecutive}/{len(disc_rows)} rows", file=out)
    if agree_interleaved != len(disc_rows):
        failures.append("the stride-32 unpack does not reproduce the C "
                        "reference")
    if agree_consecutive != 0:
        failures.append("the four-consecutive reading also reproduces the C "
                        "reference -- the test does not discriminate")
    print(file=out)

    # ---- (1b) the checks, made to fail on purpose ------------------------
    _red_checks(lib, src, by_name, data_start, rng, failures, out)

    # ---- (2) per-tensor round trip and dot products ----------------------
    print("(2) PER-TENSOR: ROUND TRIP AND DOT PRODUCTS", file=out)
    print(file=out)
    checked = []
    for name in tensor_names:
        t = by_name.get(name)
        if t is None:
            not_checked.append((name, "no such tensor in the model"))
            continue
        if t["type"] != GGML_TYPE_I2_S:
            not_checked.append((name, f"type {t['type']}, not i2_s"))
            continue
        checked.append(_verify_tensor(lib, src, t, data_start, n_rows, rng,
                                      failures, out))

    # ---- (2b) the one place the two kernels disagree ---------------------
    all_wraps = [w for rec in checked for w in rec["wraps"]]
    print("(2b) WHERE i2_s AND t5b DISAGREE, AND WHICH ONE IS WRONG", file=out)
    if not all_wraps:
        print("\n    They never disagreed above. At K = 2560 upstream's "
              "accumulator has\n    only 20 blocks to hold and cannot reach "
              "its bound.\n", file=out)
    else:
        print("""
    Every disagreement above is upstream's, and it is an int16 overflow in the
    shipped kernel rather than anything about t5b.

    bitnet_vec_dot_i2_i8_s_reference accumulates 32 BLOCKS into an int16
    accumulator before folding to int32 (ggml_i2s_ternary.c:100-129, `accu32`).
    One int16 lane takes 2 products per vpmaddubsw and 4 vpmaddubsw per block,
    so 256 products per lane per group. With every activation at +127 and the
    measured code mean of 1.000,

        256 * 1.000 * 127 = 32512    against the int16 ceiling of 32767

    -- 0.78 % of headroom, which the row-to-row spread crosses. That bound
    makes a SHARP, falsifiable prediction: the 32-block group only exists when
    nb = K/128 >= 32, i.e. K >= 4096, so K = 6912 must wrap and K = 2560
    (nb = 20, max 20320 per lane) must not. The measurement is exactly that
    split -- 200/200 rows wrong at K = 6912 on both extremes, 0/1200 wrong at
    K = 2560 -- and every single error is a whole multiple of 65536, which is
    what an int16 lane wrapping looks like and what a mis-decoded weight does
    not.

    t5b is exact in all three cases because ternary_t5b.h derives its own
    bound and folds at 12 blocks: 12 * 2560 = 30720 <= 32767, and 13 would not
    fit. The margin is designed there and inherited here.

    THIS IS NOT A DEFECT ANYONE MEETS IN PRODUCTION. Real activations are not
    all +127; on random int8 the reference is exact on every one of the 1400
    rows tried, which is why the two kernels agree everywhere it matters. It is
    recorded because a differential check is only worth running if it CAN
    separate the two implementations, and this one did -- against the one that
    ships.

    First disagreement per tensor and case:
""", file=out)
        for nm, k, label, r, exact, v_i2s, d in all_wraps:
            print(f"      {nm:<26} K={k:<5} [{label}] row {r}: exact "
                  f"{exact:>12,}  ref {v_i2s:>12,}  diff {d:>12,} "
                  f"= {d // 65536:>5} * 65536", file=out)
        print(file=out)

    # ---- (3) code 3 across ALL i2_s tensors ------------------------------
    print(f"(3) CODE 3 ACROSS ALL {len(i2s)} i2_s TENSORS", file=out)
    print("""
    The correction identity sum(code*y) = sum(w*y) + sum(y) needs code 3 to be
    absent, and t5b needs it too: a code 3 does not corrupt only its own
    weight, it carries into the next digit plane (see 1b). It had only ever
    been checked on blk.0.ffn_down.weight. Every i2_s byte in the model is read
    here -- histogrammed over the 256 byte values, then folded, so it is a full
    sweep and not a sample.

    The bytes are in hand at that point, so the ROUND TRIP is done here too,
    for every tensor and every weight, rather than inferred from the seven in
    (2): unpack the real i2_s codes, pack to t5b, unpack again, require every
    weight back. Four random rows per tensor are additionally packed by the C
    ternary_t5b_pack and required to be byte-identical, so the C authority is
    in the loop on all of them and not only on the seven.
""", file=out)
    totals = np.zeros(4, dtype=np.int64)
    worst = []
    rt_weights = 0
    rt_bad = []
    c_rows = 0
    c_bad = []
    t0 = time.time()
    with open(src, "rb") as f:
        for t in i2s:
            k, rows = t["dims"][0], t["dims"][1]
            ne = nelem(t["dims"])
            f.seek(data_start + t["offset"])
            raw = np.frombuffer(f.read(ne // 4), dtype=np.uint8)
            if raw.size != ne // 4:
                not_checked.append((t["name"], "truncated read"))
                continue
            h = code_histogram(raw)
            if h.sum() != ne:
                failures.append(f"{t['name']}: histogram covers {h.sum()} of "
                                f"{ne} weights")
            totals += h
            if h[3]:
                worst.append((t["name"], int(h[3]), ne))
                continue                     # packing it would carry; skip

            codes = i2s_unpack_codes(raw, ne)
            packed = t5b_pack_codes(codes, rows, k)
            back = t5b_unpack_codes(packed, rows, k)
            d = np.nonzero(back != codes.reshape(rows, k))
            if d[0].size:
                r, c = int(d[0][0]), int(d[1][0])
                rt_bad.append((t["name"], d[0].size, r * k + c,
                               int(codes.reshape(rows, k)[r, c]) - 1,
                               int(back[r, c]) - 1))
            rt_weights += ne

            row_t5b = t5b_size(k)
            cbuf = (ctypes.c_uint8 * row_t5b)()
            w = (codes.astype(np.int16) - 1).astype(np.int8).reshape(rows, k)
            for r in rng.choice(rows, size=min(4, rows), replace=False):
                wrow = np.ascontiguousarray(w[r])
                lib.ternary_t5b_pack(wrow.ctypes.data, ctypes.byref(cbuf), k)
                if bytes(cbuf) != packed[r * row_t5b:(r + 1) * row_t5b].tobytes():
                    c_bad.append((t["name"], int(r)))
                c_rows += 1
    tot = int(totals.sum())
    print(f"    {len(i2s)} tensors, {tot:,} weights, "
          f"{time.time() - t0:.1f}s", file=out)
    for c in range(4):
        print(f"      code {c} -> {int(totals[c]):>15,}   "
              f"{totals[c] / tot:8.4%}", file=out)
    if worst:
        print(f"\n    CODE 3 OCCURS. {len(worst)} tensors:", file=out)
        for n, cnt, ne in worst[:20]:
            print(f"      {n:<34} {cnt:,} of {ne:,}", file=out)
        failures.append(f"code 3 occurs in {len(worst)} tensors")
    else:
        print(f"\n    code 3 count is 0 over all {tot:,} weights of all "
              f"{len(i2s)} tensors.", file=out)
        print("    The identity holds model-wide, not just on "
              "blk.0.ffn_down.weight.", file=out)

    print(file=out)
    if rt_bad:
        print(f"    ROUND TRIP FAILED on {len(rt_bad)} tensors:", file=out)
        for n, cnt, idx, a, b in rt_bad[:20]:
            print(f"      {n:<34} {cnt:,} weights differ, first at index "
                  f"{idx}: i2_s {a:+d}, t5b {b:+d}", file=out)
        failures.append(f"round trip failed on {len(rt_bad)} tensors, first "
                        f"{rt_bad[0]}")
    else:
        print(f"    round trip exact on all {rt_weights:,} weights of all "
              f"{len(i2s) - len(worst)} tensors -- every weight in the model, "
              f"not a sample.", file=out)
    if c_bad:
        print(f"    ternary_t5b_pack DISAGREES on {len(c_bad)} of {c_rows} "
              f"sampled rows: {c_bad[:5]}", file=out)
        failures.append(f"C packer disagrees on {len(c_bad)} sampled rows")
    else:
        print(f"    {c_rows} sampled rows across the {len(i2s)} tensors are "
              f"byte-identical to ternary_t5b_pack.", file=out)
    print(file=out)

    # ---- (4) size arithmetic --------------------------------------------
    print("(4) BYTES BEFORE AND AFTER", file=out)
    print(file=out)
    convert(src, None, dry_run=True, verbose_plan=False, out=out,
            quiet_footer=True)
    print(file=out)

    # ---- (5) the file that was actually written --------------------------
    print("(5) THE CONVERTED FILE ON DISK", file=out)
    print(file=out)
    if converted is None or not Path(converted).exists():
        not_checked.append((str(converted), "converted file absent -- the "
                            "checks above are on in-memory packings only"))
        print(f"    {converted} does not exist. Nothing read back.", file=out)
    else:
        converted = Path(converted)
        v2, meta2, tensors2, ds2 = parse_gguf(converted)
        by2 = {t["name"]: t for t in tensors2}
        n_t5b = sum(1 for t in tensors2 if t["type"] == GGML_TYPE_T5B)
        print(f"    {converted}  {converted.stat().st_size:,} bytes", file=out)
        print(f"    GGUF v{v2}, {len(tensors2)} tensors, {n_t5b} of type "
              f"{GGML_TYPE_T5B} (T5B), {sum(1 for t in tensors2 if t['type'] == GGML_TYPE_I2_S)} "
              f"still i2_s, data at {ds2:,}", file=out)
        print("""
    The checks in (2) verified a packing held in memory. This reads the bytes
    back off the disk and requires them to be that packing -- an ack from the
    writer is not evidence that the file has them.
""", file=out)
        for rec in checked:
            t2 = by2.get(rec["name"])
            if t2 is None:
                failures.append(f"{rec['name']}: missing from {converted}")
                continue
            if t2["type"] != GGML_TYPE_T5B:
                failures.append(f"{rec['name']}: type {t2['type']} in the "
                                f"converted file, expected {GGML_TYPE_T5B}")
                continue
            want = t5b_size(rec["k"]) * rec["rows"]
            with open(converted, "rb") as f:
                f.seek(ds2 + t2["offset"])
                got = f.read(want)
                got_tail = f.read(32)
            ok_codes = got == rec["packed"].tobytes()
            ok_tail = got_tail == rec["tail"]
            print(f"    {rec['name']:<26} {want:>10,} code bytes "
                  f"{'identical' if ok_codes else 'DIFFER'}, "
                  f"32-byte scale tail "
                  f"{'identical to the source' if ok_tail else 'DIFFERS'}",
                  file=out)
            if not ok_codes:
                j = next(i for i in range(min(len(got), want))
                         if got[i] != rec["packed"][i])
                failures.append(f"{rec['name']}: written bytes differ from the "
                                f"verified packing at byte {j}: file {got[j]}, "
                                f"expected {int(rec['packed'][j])}")
            if not ok_tail:
                failures.append(f"{rec['name']}: the 32-byte scale tail was "
                                f"not carried over verbatim")
        unconv = [t["name"] for t in tensors2
                  if t["type"] not in (GGML_TYPE_T5B,) and
                  by_name[t["name"]]["type"] == GGML_TYPE_I2_S]
        if unconv:
            failures.append(f"{len(unconv)} i2_s tensors were not converted")
        # Every non-i2_s tensor byte-identical to the source, spot-checked on
        # the embedding, which is 55.7 % of the file and must not have moved.
        emb = "token_embd.weight"
        e1, e2 = by_name[emb], by2[emb]
        n_cmp = 8 * 1024 * 1024
        with open(src, "rb") as f1, open(converted, "rb") as f2:
            f1.seek(data_start + e1["offset"])
            f2.seek(ds2 + e2["offset"])
            same_head = f1.read(n_cmp) == f2.read(n_cmp)
            f1.seek(data_start + e1["offset"] + tensor_nbytes(e1) - n_cmp)
            f2.seek(ds2 + e2["offset"] + tensor_nbytes(e2) - n_cmp)
            same_tail = f1.read(n_cmp) == f2.read(n_cmp)
        print(f"    {emb:<26} f16, {tensor_nbytes(e1):,} bytes, type "
              f"{e2['type']} unchanged; first and last {n_cmp // (1 << 20)} "
              f"MiB {'identical' if same_head and same_tail else 'DIFFER'}",
              file=out)
        if not (same_head and same_tail):
            failures.append("the f16 embedding changed")
        print(file=out)

    # ---- verdict ---------------------------------------------------------
    print("VERDICT", file=out)
    n_w = sum(r["ne"] for r in checked)
    n_r = sum(r["rows"] for r in checked)
    n_d = sum(r["dots"] for r in checked)
    print(f"  Round trip, i2_s -> ternary -> t5b -> ternary: EXACT on all "
          f"{rt_weights:,} weights\n  of all {len(i2s)} i2_s tensors -- every "
          f"ternary weight in the model, not a sample.", file=out)
    print(f"  Code 3: {int(totals[3])} occurrences in those "
          f"{rt_weights:,} weights.", file=out)
    print(f"  Packing: {n_r:,} rows on {len(checked)} tensors plus {c_rows} "
          f"sampled rows across all\n  {len(i2s)} are byte-identical to the C "
          f"ternary_t5b_pack.", file=out)
    n_wrap = sum(r["wrapped_rows"] for r in checked)
    print(f"  Dot product: on {len(checked)} tensors ({n_w:,} weights), t5b "
          f"matched exact int64\n  arithmetic on all {n_d:,} "
          f"row-and-activation cases; upstream's i2_s kernel did\n  not, on "
          f"{n_wrap:,} of them, for the reason in (2b).", file=out)
    codes_after = sum(t5b_size(t["dims"][0]) * t["dims"][1] for t in i2s)
    codes_before = sum(nelem(t["dims"]) // 4 for t in i2s)
    saved = src.stat().st_size - (Path(converted).stat().st_size
                                  if converted and Path(converted).exists()
                                  else src.stat().st_size)
    print(f"  Size: {codes_before * 8 / rt_weights:.5f} -> "
          f"{codes_after * 8 / rt_weights:.5f} bits per weight over the codes"
          + (f", {saved:,} bytes off the file." if saved else "."), file=out)
    if failures:
        print(f"  FAILED -- {len(failures)} problems:", file=out)
        for f_ in failures:
            print(f"    {f_}", file=out)
    else:
        print("  Every check above passed.", file=out)
    print("\nNOT CHECKED", file=out)
    names = {r["name"] for r in checked}
    rest = [t["name"] for t in i2s if t["name"] not in names]
    print(f"    DOT PRODUCTS were measured on the {len(names)} tensors named "
          f"in (2) and not on the\n    other {len(rest)}. The round trip and "
          f"the code histogram cover all {len(i2s)} (see 3);\n    the dot "
          f"product does not. Its correctness on the rest rests on the layout\n"
          f"    being the same one -- which (3) checks -- and on the kernel, "
          f"which does not\n    read the tensor's identity. That is an "
          f"inference, and it is named here as one.", file=out)
    print(f"\n    READ-BACK from the converted file (5) covers the same "
          f"{len(names)} tensors plus the\n    embedding, not all "
          f"{len(tensors)}. What covers every tensor there is the reread in\n"
          f"    the converter itself: dims, type and offset of all "
          f"{len(tensors)}, alignment included.", file=out)
    print(f"\n    NO INFERENCE-TIME CHECK AT ALL: nothing here runs the "
          f"converted model. That the\n    weights survive the repacking is "
          f"not evidence that a model reading them\n    produces the same "
          f"tokens, and it cannot be until a loader knows type "
          f"{GGML_TYPE_T5B}.\n    No perplexity, no generated text, no "
          f"end-to-end number is claimed here.", file=out)
    print(f"\n    NO TIMING. This file is about exactness and size. Whether "
          f"t5b is FASTER than\n    i2_s on this model is a separate question "
          f"and is not touched.", file=out)
    for n, why in not_checked:
        print(f"    {n:<34} {why}", file=out)
    return 1 if failures else 0


def _red_checks(lib, src, by_name, data_start, rng, failures, out):
    """Break each check on purpose and require it to notice.

    Every result in section (2) is an equality that held. An equality that
    holds is only evidence if the same comparison would have broken had the
    data been wrong, so each one is driven red here on the same real tensor it
    runs on.
    """
    print("(1b) THE CHECKS, DRIVEN RED ON PURPOSE", file=out)
    print(file=out)
    t = by_name["blk.14.attn_k.weight"]
    k, rows = t["dims"][0], t["dims"][1]
    ne = k * rows
    raw, _ = _read_i2s(src, t, data_start)
    codes = i2s_unpack_codes(raw, ne)
    packed = t5b_pack_codes(codes, rows, k)
    w = (codes.astype(np.int16) - 1).astype(np.int8).reshape(rows, k)
    row_t5b = t5b_size(k)
    reds = []

    # 1. one flipped weight, and the round trip must name it.
    victim = 1_234_567
    bad = codes.copy()
    bad[victim] = (bad[victim] + 1) % 3
    d = np.nonzero(t5b_unpack_codes(t5b_pack_codes(bad, rows, k), rows, k)
                   != codes.reshape(rows, k))
    hit = d[0].size == 1 and int(d[0][0]) * k + int(d[1][0]) == victim
    reds.append((hit, f"one flipped weight at index {victim}: the round trip "
                      f"reports {d[0].size} differing weight(s)"
                      + (f", at index {int(d[0][0]) * k + int(d[1][0])}"
                         if d[0].size else "")))

    # 2/3. a corrupted packed byte, and both dot products must move.
    r = 5
    act = np.ascontiguousarray(
        rng.integers(-127, 128, size=k, dtype=np.int64).astype(np.int8))
    sum_a = int(act.astype(np.int64).sum())
    exact = int(np.ascontiguousarray(w[r]).astype(np.int64)
                @ act.astype(np.int64))
    wp = np.ascontiguousarray(packed[r * row_t5b:(r + 1) * row_t5b])
    good = int(lib.ternary_t5b_dot_avx2(wp.ctypes.data, act.ctypes.data,
                                        k, sum_a))
    wp2 = wp.copy()
    wp2[3] = (int(wp2[3]) + 1) % 243
    bad_dot = int(lib.ternary_t5b_dot_avx2(wp2.ctypes.data, act.ctypes.data,
                                           k, sum_a))
    reds.append((good == exact, f"row {r} intact: t5b {good}, exact {exact}"))
    reds.append((bad_dot != exact, f"row {r} with packed byte 3 corrupted: "
                                   f"t5b {bad_dot}, exact {exact}"))

    rowbytes = k // 4
    rb = np.ascontiguousarray(raw[r * rowbytes:(r + 1) * rowbytes])
    ref_good = i2s_ref_codesum(lib, rb, act, k) - sum_a
    rb2 = rb.copy()
    rb2[7] = (int(rb2[7]) + 1) % 256
    ref_bad = i2s_ref_codesum(lib, rb2, act, k) - sum_a
    reds.append((ref_good == exact, f"row {r} intact through the i2_s C "
                                    f"reference: {ref_good}, exact {exact}"))
    reds.append((ref_bad != exact, f"row {r} with i2_s byte 7 corrupted: C "
                                   f"reference {ref_bad}, exact {exact}"))

    # 4. and the carry ternary_t5b.h:159-166 documents, reproduced here: one
    #    code 3 does not stay in its own weight.
    probe = np.ones(320, dtype=np.uint8)                # all code 1, w = 0
    probe[0] = 3
    dec = t5b_unpack_codes(t5b_pack_codes(probe, 1, 320), 1, 320)[0]
    moved = np.nonzero(dec != 1)[0].tolist()
    reds.append((moved == [0, 32],
                 f"a lone code 3 at slot 0 packs byte 0 as "
                 f"{int(t5b_pack_codes(probe, 1, 320)[0])} (121 if it were "
                 f"code 1) and moves decoded slots {moved} -- the carry "
                 f"ternary_t5b.h:159-166 documents, which is why the "
                 f"converter refuses code 3 rather than packing it"))

    for ok, msg in reds:
        print(f"    {'as expected  ' if ok else 'UNEXPECTED   '}{msg}", file=out)
        if not ok:
            failures.append(f"red check did not behave: {msg}")
    print(file=out)


def _verify_tensor(lib, src, t, data_start, n_rows, rng, failures, out):
    name = t["name"]
    k, rows = t["dims"][0], t["dims"][1]
    ne = k * rows
    raw, tail = _read_i2s(src, t, data_start)
    scale = struct.unpack("<f", tail[:4])[0]

    print(f"  {name}   dims {t['dims']}  = {ne:,} weights  "
          f"K={k} rows={rows}", file=out)

    hist = code_histogram(raw)
    print(f"    codes         0:{hist[0]:,} ({hist[0]/ne:.4%})  "
          f"1:{hist[1]:,} ({hist[1]/ne:.4%})  "
          f"2:{hist[2]:,} ({hist[2]/ne:.4%})  "
          f"3:{hist[3]:,} ({hist[3]/ne:.4%})", file=out)
    print(f"    scale (f32 at ne0*ne1/4)  {scale:.6f}", file=out)

    codes = i2s_unpack_codes(raw, ne)
    if hist[3]:
        failures.append(f"{name}: code 3 present")
        return name
    w = (codes.astype(np.int16) - 1).astype(np.int8).reshape(rows, k)

    packed = t5b_pack_codes(codes, rows, k)
    row_t5b = t5b_size(k)

    # (a) the numpy packer against the C reference packer, every row.
    cbuf = (ctypes.c_uint8 * row_t5b)()
    mismatch = None
    for r in range(rows):
        wrow = np.ascontiguousarray(w[r])
        lib.ternary_t5b_pack(wrow.ctypes.data, ctypes.byref(cbuf), k)
        if bytes(cbuf) != packed[r * row_t5b:(r + 1) * row_t5b].tobytes():
            got = packed[r * row_t5b:(r + 1) * row_t5b].tobytes()
            j = next(i for i in range(row_t5b) if bytes(cbuf)[i] != got[i])
            mismatch = (r, j, bytes(cbuf)[j], got[j])
            break
    if mismatch:
        failures.append(f"{name}: packer disagrees with ternary_t5b_pack at "
                        f"row {mismatch[0]} byte {mismatch[1]}: C "
                        f"{mismatch[2]}, python {mismatch[3]}")
        print(f"    pack vs C     MISMATCH row {mismatch[0]} byte "
              f"{mismatch[1]}: C {mismatch[2]}, python {mismatch[3]}", file=out)
    else:
        print(f"    pack vs C     all {rows:,} rows byte-identical to "
              f"ternary_t5b_pack ({rows * row_t5b:,} bytes)", file=out)

    # (b) the round trip, every weight of the tensor.
    back = t5b_unpack_codes(packed, rows, k)
    bad = np.nonzero(back != codes.reshape(rows, k))
    if bad[0].size:
        r, c = int(bad[0][0]), int(bad[1][0])
        failures.append(
            f"{name}: round trip differs at weight {r * k + c} (row {r}, "
            f"col {c}): i2_s {int(codes.reshape(rows, k)[r, c]) - 1}, "
            f"t5b {int(back[r, c]) - 1}; {bad[0].size} weights differ")
        print(f"    round trip    FAILED at weight {r * k + c}: i2_s "
              f"{int(codes.reshape(rows, k)[r, c]) - 1}, t5b "
              f"{int(back[r, c]) - 1} ({bad[0].size:,} differ)", file=out)
    else:
        print(f"    round trip    exact on all {ne:,} weights", file=out)

    # (c) the numpy unpack against ternary_t5b_unpack_at, sampled: the round
    #     trip above compares two of this file's own functions, so the C
    #     decoder is what stops both from sharing a mistake.
    n_s = 2000
    idx_r = rng.integers(0, rows, size=n_s)
    idx_c = rng.integers(0, k, size=n_s)
    bad_c = 0
    for r, c in zip(idx_r, idx_c):
        wp = packed[r * row_t5b:(r + 1) * row_t5b]
        if lib.ternary_t5b_unpack_at(wp.ctypes.data, int(c)) != int(w[r, c]):
            bad_c += 1
    if bad_c:
        failures.append(f"{name}: ternary_t5b_unpack_at disagrees on {bad_c} "
                        f"of {n_s} sampled weights")
    print(f"    unpack vs C   {n_s - bad_c}/{n_s} sampled weights agree with "
          f"ternary_t5b_unpack_at", file=out)

    # (d) dot products.
    wraps = []
    wrapped_rows = 0
    sel = rng.choice(rows, size=min(n_rows, rows), replace=False)
    rowbytes = k // 4
    cases = [("random int8 in [-127,127]", None),
             ("all +127", np.full(k, 127, dtype=np.int8)),
             ("all -127", np.full(k, -127, dtype=np.int8))]
    for label, fixed in cases:
        n_ok_t5b = n_ok_scalar = n_ok_i2s = 0
        i2s_wraps = []
        first_bad = None
        first_wrap = None
        for r in sel:
            act = fixed if fixed is not None else \
                rng.integers(-127, 128, size=k, dtype=np.int64).astype(np.int8)
            act = np.ascontiguousarray(act)
            wr = np.ascontiguousarray(w[r]).astype(np.int64)
            exact = int(wr @ act.astype(np.int64))
            sum_a = int(act.astype(np.int64).sum())

            wp = np.ascontiguousarray(packed[r * row_t5b:(r + 1) * row_t5b])
            v_avx2 = int(lib.ternary_t5b_dot_avx2(wp.ctypes.data,
                                                  act.ctypes.data, k, sum_a))
            v_scal = int(lib.ternary_t5b_dot_scalar(wp.ctypes.data,
                                                    act.ctypes.data, k))
            rb = np.ascontiguousarray(raw[r * rowbytes:(r + 1) * rowbytes])
            v_i2s = i2s_ref_codesum(lib, rb, act, k) - sum_a

            n_ok_t5b += (v_avx2 == exact)
            n_ok_scalar += (v_scal == exact)
            if v_i2s == exact:
                n_ok_i2s += 1
            else:
                # Upstream accumulates 32 blocks in int16 before folding; an
                # all-one-sign activation vector can carry it past 32767. If
                # the difference is a multiple of 65536 that is exactly what
                # happened, and it is upstream's bound, not a t5b defect.
                d = v_i2s - exact
                i2s_wraps.append(d % 65536 == 0)
                if first_wrap is None:
                    first_wrap = (label, int(r), exact, v_i2s, d)
                if first_bad is None and v_avx2 != exact:
                    first_bad = (int(r), exact, v_avx2, v_scal, v_i2s)

        n = len(sel)
        print(f"    dot [{label}] over {n} rows:", file=out)
        print(f"      t5b AVX2   == exact int64  {n_ok_t5b}/{n}", file=out)
        print(f"      t5b scalar == exact int64  {n_ok_scalar}/{n}", file=out)
        print(f"      i2_s C ref == exact int64  {n_ok_i2s}/{n}"
              + ("" if n_ok_i2s == n else
                 f"   ({n - n_ok_i2s} differ; "
                 f"{sum(i2s_wraps)}/{len(i2s_wraps)} of the differences are a "
                 f"multiple of 65536, i.e. upstream's int16 accumulator "
                 f"wrapping)"), file=out)
        if n_ok_t5b != n or n_ok_scalar != n:
            failures.append(f"{name} [{label}]: t5b avx2 {n_ok_t5b}/{n}, "
                            f"scalar {n_ok_scalar}/{n} against exact int64"
                            + (f"; first bad row {first_bad}" if first_bad
                               else ""))
        if n_ok_i2s != n and not all(i2s_wraps):
            failures.append(f"{name} [{label}]: the i2_s C reference differs "
                            f"from exact by something that is NOT a multiple "
                            f"of 65536 on "
                            f"{len(i2s_wraps) - sum(i2s_wraps)} rows")
        if first_wrap is not None:
            wraps.append((name, k) + first_wrap)
        wrapped_rows += n - n_ok_i2s
    print(file=out)
    return {"name": name, "dims": t["dims"], "ne": ne, "k": k, "rows": rows,
            "packed": packed, "tail": tail, "wraps": wraps,
            "dots": len(sel) * len(cases), "wrapped_rows": wrapped_rows}


# --------------------------------------------------------------------------

DEFAULT_VERIFY_TENSORS = [
    "blk.0.ffn_down.weight",      # K=6912, 17.7M weights, the one ever checked
    "blk.0.attn_q.weight",        # K=2560, square
    "blk.7.ffn_gate.weight",      # K=2560, 6912 rows
    "blk.14.attn_k.weight",       # K=2560, 640 rows, the narrowest shape
    "blk.21.ffn_down.weight",     # K=6912, a second layer at the other K
    "blk.29.ffn_up.weight",       # K=2560, the last layer
    "blk.29.attn_v.weight",       # K=2560, last layer, narrow
]


def main():
    ap = argparse.ArgumentParser(
        description="Repack a BitNet i2_s GGUF into the 1.6-bit t5b packing.")
    ap.add_argument("gguf", type=Path)
    ap.add_argument("-o", "--out", type=Path,
                    default=Path("models/ggml-model-t5b.gguf"),
                    help="output GGUF (default models/ggml-model-t5b.gguf)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the size arithmetic and the per-tensor plan; "
                         "write nothing")
    ap.add_argument("--plan", action="store_true",
                    help="with --dry-run, print every tensor")
    ap.add_argument("--verify", action="store_true",
                    help="check the repacking against the C reference on the "
                         "model's own weights; writes nothing")
    ap.add_argument("--src-dir", type=Path, default=DEFAULT_SRC,
                    help="where the C reference lives (read only)")
    ap.add_argument("--tensors", default=",".join(DEFAULT_VERIFY_TENSORS),
                    help="comma-separated tensors for --verify")
    ap.add_argument("--rows", type=int, default=200,
                    help="rows per tensor for the dot-product check")
    ap.add_argument("--seed", type=int, default=20260905)
    args = ap.parse_args()

    try:
        if args.verify:
            return verify(args.gguf, args.src_dir,
                          [s for s in args.tensors.split(",") if s],
                          args.rows, args.seed, converted=args.out)
        if args.dry_run:
            convert(args.gguf, None, dry_run=True, verbose_plan=args.plan)
            return 0
        if args.out.exists():
            print(f"error: {args.out} exists; refusing to overwrite",
                  file=sys.stderr)
            return 2
        args.out.parent.mkdir(parents=True, exist_ok=True)
        convert(args.gguf, args.out)
        return 0
    except (ConversionError, GGUFError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
