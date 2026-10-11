# TurboQuant tools: build the Metal harnesses and run the kernel tests or the attention timing.
# They run the real Splash kernels on the GPU, so run them only when no Splash server or benchmark
# suite is using it.
#
#   make test            store, rotation and attention tests for turbo and turbo-int4 (8 RESULT lines)
#   make perf H=57344    attention time per layer for int8, turbo and turbo-int4
#   make clean
CXX := clang++
CXXFLAGS := -std=c++20 -O2 -fobjc-arc
FRAMEWORKS := -framework Foundation -framework Metal
HARNESSES := turbo_harness turbo_attn_harness attn_perf
FLATTEN := ../splash-src/dev/tools/flatten_metal.py
WORK ?= $(or $(TMPDIR),/tmp)
export TURBO_WORK := $(WORK)
H ?= 4096 16384 57344

.PHONY: all test perf clean
all: $(HARNESSES)

$(HARNESSES): %: %.mm
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) $< -o $@

# Attention cases as "committed tokens, splits": 1, 3 and 10 pages.
# Each test prints "RESULT: PASS" or "RESULT: FAIL" last; make stops at the first that does not pass.
check = out=$$($(1) 2>&1 | tail -1); echo "$(2): $$out"; case "$$out" in *PASS*) ;; *) exit 1 ;; esac

test: turbo_harness turbo_attn_harness
	@$(call check,python3 turbo_kernel_test.py,turbo store + rotation)
	@$(call check,python3 turbo_int4_test.py,turbo-int4 store + rotation)
	@for c in "24 1" "85 2" "300 3"; do for f in "" int4; do \
	  $(call check,python3 turbo_attention_test.py $$c $$f,attention $$c $${f:-turbo}); \
	done; done

perf: attn_perf
	python3 $(FLATTEN) ../splash-src/runtime perf_kernels.metal > $(WORK)/perf_kernels.flat.metal
	@for h in $(H); do ./attn_perf $(WORK)/perf_kernels.flat.metal $$h; done

clean:
	rm -f $(HARNESSES)
