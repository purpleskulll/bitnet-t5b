# bitnet-t5b -- build the test suite and the benchmarks.
#
# The flags are the paper's, not a convenience default: -O3 with AVX2 and FMA
# forced on and -march=native tuning for the measuring host. Every rate in the
# paper was produced by a binary built this way, so a build with different
# flags is measuring a different thing.
#
#   make            build everything buildable in this checkout
#   make test       build and RUN the suite -- the gate, non-zero on failure
#   make bench      build the benchmarks; skips the i2_s arm LOUDLY if the
#                   reference has not been fetched
#   make i2s-ref    fetch the i2_s baseline from the pinned upstream commit
#   make clean      remove build/
#
# WHAT NEEDS THE FETCHED REFERENCE. Three of the four benchmarks compare
# against upstream's own i2_s kernel, which this repository does not carry --
# it is MIT-licensed upstream code and is extracted from a named commit by
# tools/fetch_i2s_reference.sh instead of being redistributed here. Until that
# script has run, bench_alu, bench_threads and bench_token cannot be built.
# They are SKIPPED WITH A MESSAGE, never silently: a benchmark suite that
# quietly builds three of four targets and exits 0 is how a missing baseline
# turns into a published number that was never measured against anything.
# bench_ports is self-contained and always builds.

CC      ?= gcc
CFLAGS  ?= -O3 -mavx2 -mfma -march=native -Wall -Wextra -std=c11
LDLIBS  ?=

BIN  := build
OBJ  := $(BIN)/obj

# The generated i2_s baseline. Presence of the .c is the single condition that
# decides whether the i2_s arm is buildable; $(wildcard) evaluates it at parse
# time, which is correct here because the fetch is a separate, explicit step.
I2S_REF   := src/ggml_i2s_ternary.c
HAVE_I2S  := $(wildcard $(I2S_REF))

TESTS  := $(BIN)/test_t5b $(BIN)/test_t10
BENCH_ALWAYS := $(BIN)/bench_ports
BENCH_I2S    := $(BIN)/bench_alu $(BIN)/bench_threads $(BIN)/bench_token

ifeq ($(HAVE_I2S),)
BENCH := $(BENCH_ALWAYS)
else
BENCH := $(BENCH_ALWAYS) $(BENCH_I2S)
endif

.PHONY: all test bench i2s-ref i2s-check clean

all: $(TESTS) $(BIN)/ggml_t5b_glue.o bench

$(OBJ):
	mkdir -p $(OBJ)

# ----------------------------------------------------------------- objects ---

$(OBJ)/ternary_t5b.o: src/ternary_t5b.c src/ternary_t5b.h | $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ)/ternary_t10.o: src/ternary_t10.c src/ternary_t10.h | $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ)/ggml_i2s_ternary.o: $(I2S_REF) src/ggml_i2s_ternary.h | $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

# The ggml-facing adapter for GGML_TYPE_T5B. Built as an object and not a
# binary: what it adapts is already covered by test_t5b, and its ggml-side
# contract can only be exercised inside llama.cpp. It is in `all` so that a
# changed ternary_t5b signature is caught here in a second rather than inside a
# twenty-minute container build.
$(BIN)/ggml_t5b_glue.o: src/ggml_t5b_glue.c src/ggml_t5b_glue.h src/ternary_t5b.h | $(OBJ)
	$(CC) $(CFLAGS) -Isrc -c $< -o $@

# ------------------------------------------------------------------- tests ---

$(BIN)/test_t5b: src/test_t5b.c $(OBJ)/ternary_t5b.o | $(OBJ)
	$(CC) $(CFLAGS) -Isrc $^ -o $@ $(LDLIBS)

$(BIN)/test_t10: src/test_t10.c $(OBJ)/ternary_t10.o | $(OBJ)
	$(CC) $(CFLAGS) -Isrc $^ -o $@ $(LDLIBS)

# The gate. Every suite must pass; a non-zero exit stops the build.
test: $(TESTS)
	./$(BIN)/test_t5b
	./$(BIN)/test_t10

# -------------------------------------------------------------- benchmarks ---

# Port throughput. No project header, no model, no fetched reference: this is
# the one benchmark that measures the machine rather than the format.
$(BIN)/bench_ports: benchmarks/bench_ports.c | $(OBJ)
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

$(BIN)/bench_alu: benchmarks/bench_alu.c $(OBJ)/ggml_i2s_ternary.o \
                  $(OBJ)/ternary_t10.o $(OBJ)/ternary_t5b.o | $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

$(BIN)/bench_threads: benchmarks/bench_threads.c $(OBJ)/ggml_i2s_ternary.o | $(OBJ)
	$(CC) $(CFLAGS) -pthread $^ -o $@ $(LDLIBS)

$(BIN)/bench_token: benchmarks/bench_token.c $(OBJ)/ggml_i2s_ternary.o \
                    $(OBJ)/ternary_t10.o $(OBJ)/ternary_t5b.o | $(OBJ)
	$(CC) $(CFLAGS) -pthread $^ -o $@ $(LDLIBS)

bench: $(BENCH)
ifeq ($(HAVE_I2S),)
	@echo ''
	@echo '  ####################################################################'
	@echo '  #  SKIPPED: bench_alu, bench_threads, bench_token                  #'
	@echo '  #                                                                  #'
	@echo '  #  These three measure this format AGAINST upstream i2_s, and      #'
	@echo '  #  $(I2S_REF) is not present.                          #'
	@echo '  #  It is MIT-licensed upstream code and is not redistributed       #'
	@echo '  #  here -- see NOTICE. Fetch it from the pinned commit with        #'
	@echo '  #                                                                  #'
	@echo '  #      make i2s-ref                                                #'
	@echo '  #                                                                  #'
	@echo '  #  or, offline, from a checkout you already have:                  #'
	@echo '  #                                                                  #'
	@echo '  #      tools/fetch_i2s_reference.sh --from <BitNet-checkout>       #'
	@echo '  #                                                                  #'
	@echo '  #  Built: bench_ports only. Any comparison against i2_s made from  #'
	@echo '  #  this build would have no baseline in it.                        #'
	@echo '  ####################################################################'
	@echo ''
else
	@echo 'built all four benchmarks (i2_s reference present)'
endif

# ------------------------------------------------------------- the fetcher ---

i2s-ref:
	tools/fetch_i2s_reference.sh

# Exits non-zero if the reference is absent. For use in a script that must not
# proceed without a baseline.
i2s-check:
	tools/fetch_i2s_reference.sh --check

clean:
	rm -rf $(BIN)
