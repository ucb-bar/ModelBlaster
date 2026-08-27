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
CFLAGS    ?= -O2 -static
KERNEL_CFLAGS ?=
INCLUDES  := -I src -I $(MODEL_DIR)

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
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
$(OBJDIR)/model.o: $(MODEL_DIR)/model.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
$(OBJDIR)/weights.o: $(MODEL_DIR)/weights.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@
$(OBJDIR)/buffers.o: $(MODEL_DIR)/buffers.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Only kernels.c gets the backend ISA flags.
$(OBJ_KERNELS): $(MODEL_DIR)/kernels.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(KERNEL_CFLAGS) $(INCLUDES) -c $< -o $@

# test_io.S .incbin's the golden/input blobs; it must assemble from MODEL_DIR
# so the relative .incbin paths resolve.
$(OBJ_TESTIO): $(MODEL_DIR)/test_io.S | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OUT): $(OBJS_PLAIN) $(OBJ_KERNELS) $(OBJ_TESTIO)
	@mkdir -p $(dir $(OUT))
	$(CC) $(CFLAGS) $^ -o $@ -lm
	@echo "built $(OUT)"

clean:
	rm -rf $(OBJDIR) $(OUT)
