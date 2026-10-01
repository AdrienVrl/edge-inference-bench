# Build baseline (naive/autovec/neon - runs on any AArch64 core):
#   make
# Build the dotprod variant too (needs Armv8.2+ dot-product support - check
# first with: grep asimddp /proc/cpuinfo):
#   make dotprod
# Both together:
#   make all-variants

CC ?= gcc
CFLAGS = -O3 -march=armv8-a

neon_bench: kernel.c
	$(CC) $(CFLAGS) -o $@ $<

neon_bench_dotprod: kernel.c
	$(CC) -O3 -march=armv8.2-a+dotprod -DENABLE_DOTPROD -o $@ $<

dotprod: neon_bench_dotprod

all-variants: neon_bench neon_bench_dotprod

clean:
	rm -f neon_bench neon_bench_dotprod

.PHONY: dotprod all-variants clean
