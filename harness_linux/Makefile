# ModelBlaster Linux harness.
#
# Plain make rather than CMake: this builds five C files and links them. The
# Zephyr path needs CMake because Zephyr does; nothing here does.
#
#   make MODEL_DIR=<generated dir> CROSS=riscv64-unknown-linux-gnu- \
#        KERNEL_CFLAGS="-march=rv64gcv_zvl256b" OUT=build/mlp
#
# KERNEL_CFLAGS are applied ONLY to kernels.c, matching the Zephyr harness --
# the backend's ISA flags have no business perturbing the harness or the
# weights blob.

MODEL_DIR ?= $(error set MODEL_DIR to a generate_skeleton output directory)
CROSS     ?=
OUT       ?= build/harness
CC        := $(CROSS)gcc
# Compiler for kernels.c ONLY. Defaults to the same cross gcc as everything
# else; override when a curated kernel needs a newer compiler than the Linux
# cross-toolchain provides. The concrete case: the RVV fp16 intrinsics
# (vfloat16m4_t, vfmacc.vv f16, vfredusum.vs f16) that the Zvfh curated kernels
# are written against do not exist in GCC 13.2, which is the only
# riscv64-unknown-linux-gnu compiler installed here; GCC >= 14 has them. Since
# kernels.c is a leaf translation unit whose only external references are libm
# (expf/tanhf), an object built by a bare-metal riscv64-unknown-elf gcc of the
# same lp64d ABI links into this glibc static binary and runs -- verified on the
# board. Kept as an explicit opt-in rather than a search: a build must not
# silently change compilers.
KERNEL_CC ?= $(CC)
CFLAGS    ?= -O2 -static
KERNEL_CFLAGS ?=
INCLUDES  := -I src -I $(MODEL_DIR)

# Auto-generated header dependencies. Without them a rule's only prerequisite
# is its .c file, so a change to the GENERATED headers -- model.h, test_io.h --
# does not rebuild the objects that #include them.
#
# That is not hypothetical. src/main.c bakes MODEL_INPUT_SIZE and
# MODEL_OUTPUT_SIZE in from $(MODEL_DIR)/model.h. Rebuilding the same model at
# a different input geometry or class count regenerates model.h, leaves
# src/main.c untouched, and relinks a STALE main.o carrying the previous
# model's sizes. Everything else in the binary is new, so it runs, profiles
# correctly, and reports the old model's tensor sizes -- and the verify loop
# then compares that many elements against a golden of the new length. The
# symptom is a large max_abs_err that looks like a numerical failure and is
# actually a buffer-length mismatch. Cost an afternoon; see the 64x96 yolov8
# build.
DEPFLAGS := -MMD -MP

OBJDIR := $(OUT).objs
SRCS_PLAIN := src/main.c $(MODEL_DIR)/model.c $(MODEL_DIR)/weights.c $(MODEL_DIR)/buffers.c
OBJS_PLAIN := $(addprefix $(OBJDIR)/,$(notdir $(SRCS_PLAIN:.c=.o)))
OBJ_KERNELS := $(OBJDIR)/kernels.o
OBJ_TESTIO  := $(OBJDIR)/test_io.o

.PHONY: all clean
all: $(OUT)

$(OBJDIR):
	@mkdir -p $(OBJDIR)

$(OBJDIR)/main.o: src/main.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@
$(OBJDIR)/model.o: $(MODEL_DIR)/model.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@
$(OBJDIR)/weights.o: $(MODEL_DIR)/weights.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@
$(OBJDIR)/buffers.o: $(MODEL_DIR)/buffers.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

# Only kernels.c gets the backend ISA flags -- and, optionally, its own
# compiler (see KERNEL_CC above). -static is dropped here because it is a link
# flag and a bare-metal KERNEL_CC would try to resolve it against newlib.
$(OBJ_KERNELS): $(MODEL_DIR)/kernels.c | $(OBJDIR)
	$(KERNEL_CC) $(filter-out -static,$(CFLAGS)) $(KERNEL_CFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

# test_io.S .incbin's the golden/input blobs; it must assemble from MODEL_DIR
# so the relative .incbin paths resolve.
$(OBJ_TESTIO): $(MODEL_DIR)/test_io.S | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Pull in the generated .d files. `-include` so a first build, with none of
# them present yet, is not an error.
-include $(OBJS_PLAIN:.o=.d) $(OBJ_KERNELS:.o=.d)

$(OUT): $(OBJS_PLAIN) $(OBJ_KERNELS) $(OBJ_TESTIO)
	@mkdir -p $(dir $(OUT))
	$(CC) $(CFLAGS) $^ -o $@ -lm
	@echo "built $(OUT)"

clean:
	rm -rf $(OBJDIR) $(OUT)
