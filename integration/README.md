# Integrating t5b into `llama.cpp`

This directory turns the format in `../src/` into a tensor type a real
`llama.cpp` build can load and run. Nothing here redistributes upstream code:
the change is shipped as a patch and as the script that applies it.

| | |
|---|---|
| `t5b-integration.patch` | the change to the six modified upstream files, as a unified diff. The thing to **read**. |
| `apply_integration.py` | applies the same hunks by anchor and copies the four new source files in. The thing to **run**. |
| `../Dockerfile.t5b` | an incremental image that runs the script over an already-configured tree and rebuilds only what changed |

## The commit this is against

`microsoft/BitNet` does not tag releases, and its `Dockerfile` clones the
default branch with `--depth 1`. There is therefore no upstream tag to name.
These are the commits the clone actually resolved to, read off the tree every
measurement in the paper was made on:

```
microsoft/BitNet       0b341e582afbf9e1011f24744b554c96a3477eb5
                       2026-07-27, "Add VibeASR.cpp release to News (#596)"
3rdparty/llama.cpp     390c307752ab78fd8189f359d6954c9ba1be74af
                       2026-07-15, "Fix build-cann workflow by adding
                       placeholder job"
```

The submodule points at `https://github.com/isHuangXin/llama.cpp.git`, branch
`release-bitnet-embedding-0.6b-270m` — a fork, **not** `ggml-org/llama.cpp`.
The `i2_s` type this work is measured against does not exist in upstream
llama.cpp, so naming `ggml-org` here would be wrong.

`t5b-integration.patch` was generated against that commit and is verified to
apply to it with `git apply --check`.

## Applying it

```bash
git clone --recursive https://github.com/microsoft/BitNet.git
git -C BitNet checkout 0b341e582afbf9e1011f24744b554c96a3477eb5
git -C BitNet submodule update --init --recursive

python3 integration/apply_integration.py BitNet
```

The script copies `ternary_t5b.{c,h}` and `ggml_t5b_glue.{c,h}` from `../src/`
into `ggml/src/ggml-cpu/`, adds them to that directory's `CMakeLists.txt`, and
applies 19 hunks across six files. Then build as BitNet's own `Dockerfile`
does:

```bash
cd BitNet && cmake -B build ... && cmake --build build --target llama-cli -j 4
```

Or read the diff and apply it by hand:

```bash
git -C BitNet apply --directory=3rdparty/llama.cpp \
    ../integration/t5b-integration.patch
cp src/ternary_t5b.{c,h} src/ggml_t5b_glue.{c,h} \
   BitNet/3rdparty/llama.cpp/ggml/src/ggml-cpu/
```

The two routes were checked to produce byte-identical trees.

## Why a script as well as a diff

A context diff is the readable artefact and the script is the durable one. The
script is anchored on text rather than on line numbers, so it survives upstream
moving a few lines; and it is idempotent, so re-running it over an
already-patched tree — which is exactly what `Dockerfile.t5b` does — is a
no-op rather than a corruption.

Both refuse loudly. Every anchor must be present **exactly once** before
anything is written:

```
PATCH FAILED: ggml.h type enum: expected exactly 1 occurrence of anchor, found 0
The tree is NOT integrated. Fix the anchor before building --
a silently unpatched tree builds and runs normally, which is the
worst possible outcome.
```

That last line is the whole design. A half-applied patch to `ggml-cpu.c` leaves
a tree that either does not compile or, worse, computes one stride the old way
and one the new way. A refusal is strictly better.

## What it changes, and why each part is necessary rather than sufficient

| file | what |
|---|---|
| `ggml/include/ggml.h` | `GGML_TYPE_T5B = 43`, `GGML_TYPE_COUNT` 43 → 44 |
| `ggml/src/ggml.c` | the type-traits table and `ggml_nbytes` |
| `ggml/src/ggml-cpu/ggml-cpu.c` | the CPU traits entry and the `mul_mat` dispatch |
| `ggml/src/ggml-cpu/llamafile/sgemm.{h,cpp}` | `llamafile_sgemm_t5b` |
| `ggml/src/ggml-cpu/CMakeLists.txt` | compiles the two new `.c` files |

The `sgemm` hunks are the ones that are easy to leave out and expensive to
leave out. With `GGML_LLAMAFILE=ON` — the default —
`ggml_compute_forward_mul_mat` returns through `llamafile_sgemm` before the
`vec_dot` fallback is ever consulted. A tree patched only in `ggml-cpu.c`
builds, runs, produces correct output, and never executes the kernel once. That
was measured, not reasoned about: the call counter stayed at zero for a whole
inference run.

## What it deliberately does not change

The `i2_s` path. It is the control arm of every A/B in the paper, and a result
measured against a moved baseline is not a result. `43` is a free slot —
`ggml.h` lists every value and the highest assigned is `GGML_TYPE_TL2 = 42`.

A stock `llama.cpp` therefore **cannot** read a t5b file: type 43 is not
reserved upstream, and `gguf.cpp` rejects it with `invalid ggml type 43`. That
is the intended failure. A metadata flag on `i2_s` would instead read
wrong-but-valid bytes and speak fluently.

## What this does not give you

A patched, built `llama.cpp` is necessary for the model-level numbers in the
paper and not sufficient. You also need the checkpoint
(`microsoft/bitnet-b1.58-2B-4T-gguf`, ~1.1 GiB, its own licence) and the
converter, `tools/gguf_to_t5b.py`. Neither the weights nor a built image are in
this repository. See the root `README.md` for the full list of what is and is
not reproducible from a clone alone.
