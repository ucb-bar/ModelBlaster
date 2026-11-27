# K3 IR runtime.
#   make host                       x86 build, reference kernels (numerics model of the C kernels)
#   make k3 CROSS=<prefix>          riscv64 static build for the K3 (IME + RVV kernels)
#   make spike CROSS=<prefix>       riscv64 static build, reference + RVV kernels only (no IME), for Spike
MBROOT  := $(abspath ..)
RT      := $(MBROOT)/runtime
COMMON  := -O2 -ffp-contract=off -fno-strict-aliasing -Wall -Wno-unused-function -I src -I $(RT)
SRCS    := src/k3ir.c src/kern_ref.c
# CROSS: prefix of a riscv64 glibc toolchain with static libs (SpacemiT toolchain v1.1.2 used for the K3)
CROSS   ?= riscv64-unknown-linux-gnu-

host: build/k3ir_host
build/k3ir_host: $(SRCS) src/kern_stub.c src/k3ir.h
	mkdir -p build && gcc $(COMMON) -o $@ $(SRCS) src/kern_stub.c -lm -lpthread

k3: build/k3ir_k3
build/k3ir_k3: $(SRCS) src/kern_opt.c src/kern_attn.c src/k3ir.h
	mkdir -p build && $(CROSS)gcc $(COMMON) -march=rv64gcv_zicbop_zicboz -mabi=lp64d -static -DMODELBLASTER_PLATFORM_LINUX \
	  -DMODELBLASTER_AI_CPUS=\"8-15\" -o $@ $(SRCS) src/kern_opt.c src/kern_attn.c $(RT)/mb_placement.c -lm -lpthread

spike: build/k3ir_spike
build/k3ir_spike: $(SRCS) src/kern_opt.c src/kern_attn.c src/k3ir.h
	mkdir -p build && $(CROSS)gcc $(COMMON) -march=rv64gcv_zicbop_zicboz -mabi=lp64d -static -DK3_NO_IME \
	  -o $@ $(SRCS) src/kern_opt.c src/kern_attn.c -lm -lpthread
