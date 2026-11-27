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

# fp16 storage (kern_f16.c, the reference wrapper's _Float16 conversions in kern_ref.c): Zfh + Zvfh, which both
# K3 core kinds implement; the int8 / fp32 sources keep their -march
F16MARCH := rv64gcv_zfh_zvfh_zicbop_zicboz
k3: build/k3ir_k3
build/k3ir_k3: $(SRCS) src/kern_opt.c src/kern_attn.c src/kern_f16.c src/k3ir.h
	mkdir -p build/o_k3 && for f in kern_ref kern_f16; do $(CROSS)gcc $(COMMON) -march=$(F16MARCH) -mabi=lp64d \
	  -c src/$$f.c -o build/o_k3/$$f.o || exit 1; done
	$(CROSS)gcc $(COMMON) -march=rv64gcv_zicbop_zicboz -mabi=lp64d -static -DMODELBLASTER_PLATFORM_LINUX \
	  -DMODELBLASTER_AI_CPUS=\"8-15\" -o $@ src/k3ir.c src/kern_opt.c src/kern_attn.c $(RT)/mb_placement.c build/o_k3/*.o -lm -lpthread

spike: build/k3ir_spike
build/k3ir_spike: $(SRCS) src/kern_opt.c src/kern_attn.c src/kern_f16.c src/k3ir.h
	mkdir -p build/o_spike && for f in kern_ref kern_f16; do $(CROSS)gcc $(COMMON) -march=$(F16MARCH) -mabi=lp64d \
	  -DK3_NO_IME -c src/$$f.c -o build/o_spike/$$f.o || exit 1; done
	$(CROSS)gcc $(COMMON) -march=rv64gcv_zicbop_zicboz -mabi=lp64d -static -DK3_NO_IME \
	  -o $@ src/k3ir.c src/kern_opt.c src/kern_attn.c build/o_spike/*.o -lm -lpthread
