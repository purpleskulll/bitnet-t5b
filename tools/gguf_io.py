#!/usr/bin/env python3
"""
gguf_io.py -- the GGUF container primitives every tool in this directory reads
through.

WHY THIS FILE EXISTS. These primitives were originally defined inside a much
larger tool belonging to a different phase of the private tree, and the public
tools imported them from there. That import is invisible in a fresh clone: the
three importers died with ModuleNotFoundError before printing a usage line, so
"every figure is recomputed by a script in this repository" was false for
tools/gguf_inspect.py, tools/gguf_to_t5b.py and tools/dead_neurons.py. The
primitives live here now, in the repository that needs them.

WHAT IS AND IS NOT HERE. Only the container reader: the type constants, the
struct formats, the cursor, and a metadata-only reader. Tensor decoding is the
business of the tool doing the decoding (tools/gguf_inspect.py::parse_gguf
parses the tensor table; tools/gguf_to_t5b.py repacks the payload). Nothing in
this file mmaps or reads tensor data.

FORMAT. GGUF v2 and v3, little-endian, as defined by ggml. Header layout:

    magic "GGUF"      4 bytes
    version           uint32   (2 or 3)
    tensor_count      uint64
    metadata_kv_count uint64
    metadata_kv[]     { key: string, type: uint32, value: <type> }
    tensor_info[]     { name, n_dims, dims[], type: uint32, offset: uint64 }
    padding to general.alignment
    tensor data

A string is a uint64 length followed by that many UTF-8 bytes -- NOT
NUL-terminated, and the length is 64-bit even in v2.
"""

import struct

__all__ = [
    "GGUFError",
    "GGUFType",
    "_Reader",
    "_SCALAR",
    "_GGUF_U8", "_GGUF_I8", "_GGUF_U16", "_GGUF_I16",
    "_GGUF_U32", "_GGUF_I32", "_GGUF_F32", "_GGUF_BOOL",
    "_GGUF_STRING", "_GGUF_ARRAY", "_GGUF_U64", "_GGUF_I64", "_GGUF_F64",
    "read_gguf_metadata",
]


class GGUFError(RuntimeError):
    """Raised for anything the container itself is wrong about.

    Distinct from ValueError on purpose: a caller that wants to say "this file
    is not a GGUF" and a caller that wants to say "this argument is nonsense"
    are different callers, and conflating them makes a truncated file look like
    a usage error.
    """
    pass


# GGUF metadata value type tags, in the order ggml assigns them.
_GGUF_U8, _GGUF_I8, _GGUF_U16, _GGUF_I16 = 0, 1, 2, 3
_GGUF_U32, _GGUF_I32, _GGUF_F32, _GGUF_BOOL = 4, 5, 6, 7
_GGUF_STRING, _GGUF_ARRAY, _GGUF_U64, _GGUF_I64, _GGUF_F64 = 8, 9, 10, 11, 12

# tag -> (struct format, width in bytes). BOOL is one byte, not four.
_SCALAR = {
    _GGUF_U8: ("<B", 1), _GGUF_I8: ("<b", 1),
    _GGUF_U16: ("<H", 2), _GGUF_I16: ("<h", 2),
    _GGUF_U32: ("<I", 4), _GGUF_I32: ("<i", 4),
    _GGUF_F32: ("<f", 4), _GGUF_BOOL: ("<B", 1),
    _GGUF_U64: ("<Q", 8), _GGUF_I64: ("<q", 8),
    _GGUF_F64: ("<d", 8),
}


class GGUFType:
    """The tag values as attributes, for callers that prefer names.

    The module-level _GGUF_* constants stay because they are what the existing
    tools import; this is the same set spelled readably.
    """
    U8, I8, U16, I16 = _GGUF_U8, _GGUF_I8, _GGUF_U16, _GGUF_I16
    U32, I32, F32, BOOL = _GGUF_U32, _GGUF_I32, _GGUF_F32, _GGUF_BOOL
    STRING, ARRAY = _GGUF_STRING, _GGUF_ARRAY
    U64, I64, F64 = _GGUF_U64, _GGUF_I64, _GGUF_F64


class _Reader:
    """A bounds-checked cursor over an in-memory GGUF header.

    Every read goes through take(), which raises rather than returning short.
    A silent short read here would surface much later as a nonsensical tensor
    offset, and the file would be blamed for what the reader did.
    """

    def __init__(self, buf):
        self.buf = buf
        self.off = 0

    def take(self, k):
        if self.off + k > len(self.buf):
            raise GGUFError(f"truncated at offset {self.off} (wanted {k} bytes)")
        out = self.buf[self.off:self.off + k]
        self.off += k
        return out

    def scalar(self, t):
        fmt, size = _SCALAR[t]
        return struct.unpack(fmt, self.take(size))[0]

    def string(self):
        n = struct.unpack("<Q", self.take(8))[0]
        return self.take(n).decode("utf-8", errors="replace")

    def value(self, t):
        if t == _GGUF_STRING:
            return self.string()
        if t == _GGUF_ARRAY:
            et = struct.unpack("<I", self.take(4))[0]
            n = struct.unpack("<Q", self.take(8))[0]
            if et == _GGUF_STRING:
                return [self.string() for _ in range(n)]
            if et == _GGUF_ARRAY:
                return [self.value(_GGUF_ARRAY) for _ in range(n)]
            if et not in _SCALAR:
                raise GGUFError(f"unknown array element type {et}")
            fmt, size = _SCALAR[et]
            raw = self.take(size * n)
            return list(struct.unpack("<" + fmt[1] * n, raw))
        if t in _SCALAR:
            return self.scalar(t)
        raise GGUFError(f"unknown value type {t}")


def read_gguf_metadata(path):
    """Returns the metadata KV dict. Reads only the header, never the tensors."""
    with open(path, "rb") as f:
        head = f.read(24)
        if len(head) < 24 or head[:4] != b"GGUF":
            raise GGUFError(f"{path}: not a GGUF file (bad magic)")
        version, = struct.unpack("<I", head[4:8])
        if version not in (2, 3):
            raise GGUFError(f"{path}: unsupported GGUF version {version}")
        # Read a generous chunk; vocabularies of 128k tokens fit comfortably.
        f.seek(0)
        buf = f.read(256 * 1024 * 1024)

    r = _Reader(buf)
    r.take(4)                                   # magic
    r.take(4)                                   # version
    struct.unpack("<Q", r.take(8))[0]           # tensor count
    n_kv = struct.unpack("<Q", r.take(8))[0]

    meta = {}
    for _ in range(n_kv):
        key = r.string()
        vtype = struct.unpack("<I", r.take(4))[0]
        meta[key] = r.value(vtype)
    return meta


if __name__ == "__main__":
    # Self-check with no model file: round-trip every scalar tag and both
    # array shapes through _Reader, then confirm the bounds check fires.
    # `python3 tools/gguf_io.py` is a real test, not a usage message.
    import sys

    def _enc_string(s):
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    checks = 0

    for tag, (fmt, size) in sorted(_SCALAR.items()):
        probe = 1 if fmt not in ("<f", "<d") else 1.0
        raw = struct.pack(fmt, probe)
        got = _Reader(raw).value(tag)
        assert got == probe, f"tag {tag}: {got!r} != {probe!r}"
        assert len(raw) == size, f"tag {tag}: width {len(raw)} != {size}"
        checks += 1

    got = _Reader(_enc_string("bitnet-b1.58")).value(_GGUF_STRING)
    assert got == "bitnet-b1.58", got
    checks += 1

    arr = (struct.pack("<I", _GGUF_U32) + struct.pack("<Q", 3)
           + struct.pack("<III", 7, 8, 9))
    got = _Reader(arr).value(_GGUF_ARRAY)
    assert got == [7, 8, 9], got
    checks += 1

    arr = (struct.pack("<I", _GGUF_STRING) + struct.pack("<Q", 2)
           + _enc_string("a") + _enc_string("bb"))
    got = _Reader(arr).value(_GGUF_ARRAY)
    assert got == ["a", "bb"], got
    checks += 1

    try:
        _Reader(b"\x01\x02").take(8)
    except GGUFError:
        checks += 1
    else:
        print("FAIL: take() past the end did not raise", file=sys.stderr)
        sys.exit(1)

    try:
        _Reader(b"").value(99)
    except GGUFError:
        checks += 1
    else:
        print("FAIL: unknown value type did not raise", file=sys.stderr)
        sys.exit(1)

    print(f"gguf_io self-check: {checks} checks passed")
