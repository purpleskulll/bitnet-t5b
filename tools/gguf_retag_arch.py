#!/usr/bin/env python3
"""Rewrite a GGUF's architecture tag, and nothing else.

WHY THIS EXISTS
---------------
The 1.600-bit format has only ever been tested on one model, and paper §6.1
says so. The obvious second test is another ternary model of the same family:
jpacifico/Aramis-2B-BitNet-b1.58-i2s-GGUF, a fine-tune with weights this
project's converter and kernel have never seen.

It does not load:

    llama_model_load: error loading model: unknown model architecture: 'bitnet-25'

That is a metadata string, not an architecture. Compared against
microsoft/bitnet-b1.58-2B-4T with tools/gguf_inspect.py's reader:

    tensor count          332 vs 332
    identical names       yes, all 332
    shape or type diffs   0
    block_count           30 vs 30
    embedding_length      2560 vs 2560
    rope.freq_base        500000.0 vs 500000.0

Every structural parameter agrees and every tensor agrees. The two files differ
by 32 bytes in total size and by the architecture string, which also prefixes
every architecture-scoped metadata key -- so llama.cpp cannot find
`bitnet-b1.58.block_count` in a file that spells it `bitnet-25.block_count`,
and refuses the file before reading a single weight.

WHAT THIS DOES, AND WHAT IT IS NOT
  It copies the file with `general.architecture` set to the new name and every
  `<old>.` key prefix renamed to `<new>.`. Tensor DATA is copied byte for byte
  and the tensor table is rewritten only where offsets move because the
  metadata block changed length.

  It is a RELABEL. It is sound here because the check above establishes the
  architectures are the same; it would be silent corruption on two models that
  merely look alike, and the tool therefore refuses to run without --i-checked
  and prints what it is asserting.

Usage:
    python3 tools/gguf_retag_arch.py IN.gguf -o OUT.gguf \\
        --from bitnet-25 --to bitnet-b1.58 --i-checked
"""

import argparse
import pathlib
import struct
import sys

# GGUF value type ids, from ggml/src/gguf.cpp.
(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)

_FIXED = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
          F32: "<f", BOOL: "<?", U64: "<Q", I64: "<q", F64: "<d"}


class Reader:
    def __init__(self, buf):
        self.b, self.p = buf, 0

    def take(self, n):
        v = self.b[self.p:self.p + n]
        if len(v) != n:
            raise EOFError("truncated GGUF")
        self.p += n
        return v

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode("utf-8")


def write_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def read_value(r, t):
    """-> (python value, a writer that re-emits it)."""
    if t == STRING:
        s = r.string()
        return s, (lambda v=s: write_string(v))
    if t == ARRAY:
        et = r.u32()
        n = r.u64()
        items = [read_value(r, et)[0] for _ in range(n)]

        def w(et=et, items=items):
            out = struct.pack("<I", et) + struct.pack("<Q", len(items))
            for it in items:
                out += (write_string(it) if et == STRING
                        else struct.pack(_FIXED[et], it))
            return out
        return items, w
    fmt = _FIXED[t]
    v = struct.unpack(fmt, r.take(struct.calcsize(fmt)))[0]
    return v, (lambda v=v, fmt=fmt: struct.pack(fmt, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--from", dest="old", required=True)
    ap.add_argument("--to", dest="new", required=True)
    ap.add_argument("--i-checked", action="store_true",
                    help="assert that the two architectures are structurally "
                         "identical; without it this tool refuses to run")
    a = ap.parse_args()

    if not a.i_checked:
        print("refusing to run without --i-checked.\n"
              "Renaming an architecture is sound only when the tensor names, "
              "shapes, types and structural parameters already agree. Compare "
              "them first (tools/gguf_inspect.py reads both) and pass the flag "
              "to say you did.", file=sys.stderr)
        return 2

    src = pathlib.Path(a.gguf)
    raw = src.read_bytes()
    r = Reader(raw)
    if r.take(4) != b"GGUF":
        print("not a GGUF", file=sys.stderr)
        return 2
    version = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()

    kvs, renamed = [], 0
    for _ in range(n_kv):
        key = r.string()
        vt = r.u32()
        val, w = read_value(r, vt)
        newkey = key
        if key.startswith(a.old + "."):
            newkey = a.new + key[len(a.old):]
            renamed += 1
        if key == "general.architecture" and val == a.old:
            val = a.new
            w = (lambda v=a.new: write_string(v))
            renamed += 1
        kvs.append((newkey, vt, w))

    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        ttype = r.u32()
        off = r.u64()
        tensors.append((name, dims, ttype, off))
    header_end = r.p

    align = 32
    for k, vt, w in kvs:
        if k == "general.alignment":
            align = struct.unpack("<I", w())[0]
    data_start = (header_end + align - 1) // align * align

    # Rebuild the header. Tensor offsets are relative to the data section, so
    # they do not move even though the header length does -- but the data
    # section's absolute start does, and that is why the whole file is rewritten
    # rather than patched in place.
    head = b"GGUF" + struct.pack("<I", version)
    head += struct.pack("<Q", n_tensors) + struct.pack("<Q", n_kv)
    for k, vt, w in kvs:
        head += write_string(k) + struct.pack("<I", vt) + w()
    for name, dims, ttype, off in tensors:
        head += write_string(name) + struct.pack("<I", len(dims))
        for d in dims:
            head += struct.pack("<Q", d)
        head += struct.pack("<I", ttype) + struct.pack("<Q", off)
    pad = (-len(head)) % align
    head += b"\x00" * pad

    out = pathlib.Path(a.out)
    with out.open("wb") as f:
        f.write(head)
        f.write(raw[data_start:])

    print(f"  architecture   {a.old!r} -> {a.new!r}")
    print(f"  keys renamed   {renamed}")
    print(f"  tensors        {n_tensors} (data copied byte for byte)")
    print(f"  data start     {data_start:,} -> {len(head):,}")
    print(f"  file           {len(raw):,} -> {out.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
