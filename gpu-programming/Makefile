# Build: make
# Override the GPU architecture if you're not on a Turing card (T1000/T4/RTX 20xx):
#   make ARCH=sm_86   (Ampere, e.g. RTX 30xx)
#   make ARCH=sm_89   (Ada, e.g. RTX 40xx)
# Check yours with: nvidia-smi --query-gpu=compute_cap --format=csv
 
ARCH ?= sm_75
NVCC ?= nvcc
CCBIN ?= g++-13
 
matmul_bench: matmul.cu
	$(NVCC) -O3 -arch=$(ARCH) -ccbin=$(CCBIN) -lcublas -o $@ $<
 
clean:
	rm -f matmul_bench *.ncu-rep
 
.PHONY: clean
