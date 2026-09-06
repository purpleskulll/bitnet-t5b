#!/usr/bin/env python3
"""
gguf_inspect.py -- list a GGUF's tensors and, for i2_s tensors, histogram the
actual 2-bit codes in their data.

Written because sampling the file at a guessed offset produced a near-uniform
distribution over all four codes, which is not what ternary weights look like.
That was a sign of reading the wrong bytes, not a finding. This parses the
tensor table and reads exactly the bytes of a named tensor.
"""

import argparse
import collections
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from derive_filler_tokens import _Reader, _SCALAR, _GGUF_STRING, _GGUF_ARRAY, GGUFError  # noqa: E402

GGML_TYPE_NAMES = {0: "f32", 1: "f16", 36: "i2_s", 37: "i8_s", 38: "tl1", 39: "tl2"}


def parse_gguf(path):
    with open(path, "rb") as f:
        head = f.read(4096)
        if head[:4] != b"GGUF":
            raise GGUFError(f"{path}: bad magic")
        version, = struct.unpack("<I", head[4:8])
        # Read enough for metadata + tensor table.
        f.seek(0)
        buf = f.read(512 * 1024 * 1024)

    r = _Reader(buf)
    r.take(4)
    r.take(4)
    n_tensors = struct.unpack("<Q", r.take(8))[0]
    n_kv = struct.unpack("<Q", r.take(8))[0]

    meta = {}
    for _ in range(n_kv):
        key = r.string()
        vtype = struct.unpack("<I", r.take(4))[0]
        meta[key] = r.value(vtype)

    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = struct.unpack("<I", r.take(4))[0]
        dims = [struct.unpack("<Q", r.take(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack("<I", r.take(4))[0]
        offset = struct.unpack("<Q", r.take(8))[0]
        tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})

    alignment = meta.get("general.alignment", 32)
    data_start = r.off
    if data_start % alignment:
        data_start += alignment - (data_start % alignment)

    return version, meta, tensors, data_start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("-n", "--histogram", metavar="TENSOR",
                    help="histogram the 2-bit codes of this i2_s tensor "
                         "(default: the first i2_s tensor found)")
    ap.add_argument("--limit-mb", type=float, default=16.0,
                    help="how much of the tensor to read (default 16 MiB)")
    args = ap.parse_args()

    version, meta, tensors, data_start = parse_gguf(args.gguf)
    print(f"GGUF v{version}, {len(tensors)} tensors, data starts at {data_start}")
    print(f"architecture: {meta.get('general.architecture')}")

    by_type = collections.Counter(t["type"] for t in tensors)
    print("\ntensor types:")
    for t, c in sorted(by_type.items()):
        print(f"  {GGML_TYPE_NAMES.get(t, t):>6} : {c} tensors")

    i2s = [t for t in tensors if t["type"] == 36]
    if not i2s:
        print("\nno i2_s tensors -- nothing to histogram")
        return 1

    print(f"\nfirst 5 i2_s tensors:")
    for t in i2s[:5]:
        print(f"  {t['name']:<40} dims={t['dims']}  offset={t['offset']}")

    target = None
    if args.histogram:
        target = next((t for t in i2s if t["name"] == args.histogram), None)
        if target is None:
            print(f"error: no i2_s tensor named {args.histogram}", file=sys.stderr)
            return 2
    else:
        # Pick the largest, so the histogram is over a lot of weights.
        target = max(i2s, key=lambda t: t["dims"][0] * t["dims"][1] if len(t["dims"]) > 1
                     else t["dims"][0])

    nelem = 1
    for d in target["dims"]:
        nelem *= d
    nbytes = nelem // 4                       # 4 weights per byte

    print(f"\nhistogramming {target['name']}  dims={target['dims']}  "
          f"={nelem:,} weights ={nbytes:,} bytes")

    read = min(int(args.limit_mb * 1024 * 1024), nbytes)
    with open(args.gguf, "rb") as f:
        f.seek(data_start + target["offset"])
        buf = f.read(read)

    h = collections.Counter()
    for b in buf:
        h[b & 3] += 1
        h[(b >> 2) & 3] += 1
        h[(b >> 4) & 3] += 1
        h[(b >> 6) & 3] += 1
    tot = sum(h.values())
    print(f"\n2-bit codes over {tot:,} weights ({read/1048576:.1f} MiB read):")
    map2bit = [-1, 0, 1, 0]
    for c in range(4):
        print(f"  code {c} -> {h[c]:>13,}  {h[c]/tot:7.3%}   "
              f"scalar-fallback map: {map2bit[c]:+d}")

    if h[3] == 0:
        print("\n  code 3 never occurs -> the {0,1,2} encoding is exact and")
        print("  sum(code*y) = sum(w*y) + sum(y) holds without correction.")
    else:
        print(f"\n  code 3 occurs {h[3]/tot:.3%} of the time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
