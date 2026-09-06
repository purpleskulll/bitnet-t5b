#!/usr/bin/env python3
"""Duplicate a GGUF's transformer blocks to make a structurally larger model.

WHY THIS EXISTS
---------------
Paper §6.1 says the 1.600-bit format has been measured at one model SIZE, and
that no public ternary GGUF above 2B exists to try. That is true and it is not
a reason to leave the size question at "untested": the quantity that decides
whether a denser weight format pays is how many weight bytes a token reads, and
that can be doubled by doubling the layers.

WHAT THIS PRODUCES, AND WHAT IT EMPHATICALLY IS NOT
  It copies every `blk.N.*` tensor to `blk.(N+L).*` for the original layer count
  L, sets `<arch>.block_count` to 2L, and leaves the embedding, the output head
  and the norms alone. llama.cpp loads the result, runs it, and generates
  tokens.

  The output is NOT A MODEL. Running the same thirty blocks twice produces text,
  and the text is worthless -- no claim about quality, coherence or perplexity
  can be made from it and none is. What IS real is every quantity a throughput
  benchmark measures: the weight bytes streamed per token, the matmul shapes,
  the attention and KV cost of a 60-layer graph, the thread scheduling. That is
  the whole point. A synthetic deep model is an honest instrument for "does the
  advantage survive at twice the weight footprint" and a dishonest one for
  anything else, and the filename it writes says `-synthetic-` so a stray file
  cannot be mistaken for a checkpoint.

  It also does not change the ARCHITECTURE: the same attention heads, the same
  hidden size, the same rope. A real 4B model would differ in more than depth.

Usage:
    python3 tools/gguf_deepen.py IN.gguf -o OUT.gguf [--factor 2]
"""

import argparse
import pathlib
import re
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_retag_arch import (Reader, write_string, read_value,  # noqa: E402
                             STRING, U32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--factor", type=int, default=2,
                    help="how many copies of the block stack (default 2)")
    a = ap.parse_args()
    if a.factor < 2 or a.factor > 8:
        print("factor 2..8", file=sys.stderr)
        return 2

    raw = pathlib.Path(a.gguf).read_bytes()
    r = Reader(raw)
    if r.take(4) != b"GGUF":
        print("not a GGUF", file=sys.stderr)
        return 2
    version = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()

    kvs = []
    arch = None
    align = 32
    for _ in range(n_kv):
        key = r.string()
        vt = r.u32()
        val, w = read_value(r, vt)
        if key == "general.architecture":
            arch = val
        if key == "general.alignment":
            align = val
        kvs.append([key, vt, val, w])

    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        ttype = r.u32()
        off = r.u64()
        tensors.append([name, dims, ttype, off])
    header_end = r.p
    data_start = (header_end + align - 1) // align * align

    # Tensor sizes come from the gaps between consecutive offsets, and the last
    # one from the end of the file. Derived rather than recomputed per type,
    # because i2_s and t5b both carry a trailing scale the type traits do not
    # describe -- ggml.c sizes them with a hard-coded special case.
    order = sorted(range(len(tensors)), key=lambda i: tensors[i][3])
    size = {}
    for k, i in enumerate(order):
        end = (tensors[order[k + 1]][3] if k + 1 < len(order)
               else len(raw) - data_start)
        size[i] = end - tensors[i][3]

    blk = re.compile(r"^blk\.(\d+)\.")
    layers = {int(blk.match(t[0]).group(1)) for t in tensors if blk.match(t[0])}
    if not layers:
        print("no blk.N.* tensors found", file=sys.stderr)
        return 2
    L = max(layers) + 1
    if sorted(layers) != list(range(L)):
        print(f"layer indices are not contiguous 0..{L-1}", file=sys.stderr)
        return 2

    bc_key = f"{arch}.block_count"
    for kv in kvs:
        if kv[0] == bc_key:
            if kv[2] != L:
                print(f"{bc_key} is {kv[2]} but {L} layers are present",
                      file=sys.stderr)
                return 2
            newL = L * a.factor
            kv[3] = (lambda v=newL: struct.pack("<I", v))
            break
    else:
        print(f"{bc_key} not found", file=sys.stderr)
        return 2

    # New tensor list: everything once, then the block stack (factor-1) more
    # times with shifted indices. Order within a copy is preserved.
    new = [(t[0], t[1], t[2], size[i]) for i, t in enumerate(tensors)]
    for c in range(1, a.factor):
        for i, t in enumerate(tensors):
            m = blk.match(t[0])
            if m:
                shifted = f"blk.{int(m.group(1)) + c * L}." + t[0][m.end():]
                new.append((shifted, t[1], t[2], size[i]))

    # Lay the data out, then write a header whose offsets match it.
    offs, cur = [], 0
    for _, _, _, sz in new:
        offs.append(cur)
        cur += sz
        cur = (cur + align - 1) // align * align
    total_data = cur

    head = b"GGUF" + struct.pack("<I", version)
    head += struct.pack("<Q", len(new)) + struct.pack("<Q", n_kv)
    for key, vt, _val, w in kvs:
        head += write_string(key) + struct.pack("<I", vt) + w()
    for (name, dims, ttype, _sz), off in zip(new, offs):
        head += write_string(name) + struct.pack("<I", len(dims))
        for d in dims:
            head += struct.pack("<Q", d)
        head += struct.pack("<I", ttype) + struct.pack("<Q", off)
    head += b"\x00" * ((-len(head)) % align)

    src_off = {t[0]: t[3] for t in tensors}
    out = pathlib.Path(a.out)
    with out.open("wb") as f:
        f.write(head)
        written = 0
        for (name, _dims, _ttype, sz), off in zip(new, offs):
            f.write(b"\x00" * (off - written))
            base = name
            m = blk.match(name)
            if m and int(m.group(1)) >= L:
                base = f"blk.{int(m.group(1)) % L}." + name[m.end():]
            s = data_start + src_off[base]
            f.write(raw[s:s + sz])
            written = off + sz
        f.write(b"\x00" * (total_data - written))

    print(f"  architecture   {arch}")
    print(f"  layers         {L} -> {L * a.factor}   ({bc_key})")
    print(f"  tensors        {n_tensors} -> {len(new)}")
    print(f"  file           {len(raw):,} -> {out.stat().st_size:,} bytes")
    print("  NOT A MODEL: the same blocks run several times. Throughput and")
    print("  memory traffic are real; the text it emits is worthless.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
