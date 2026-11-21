# kernels/rvv_k3_a100

Hand-tuned kernels for the SpaceMiT K3 A100 cores (CPUs 8-15, VLEN=1024),
backend `rvv_k3_a100` (`-march=rv64gcv_zvl1024b_zfh_zvfh`).

File naming: `rvv_k3_a100_<op>_<algo>.c`. The picker probes this directory
first; any op without a file here falls back to the shared `kernels/rvv/`
kernels (backend alias `rvv`), recompiled at zvl1024b.

Code built for this backend is illegal on the X100 (VLEN=256): run it only
on harts 8-15.

The simple-layer kernels (silu, elu, maxpool, upsample, cat; k3-work TASK-017)
have the same text as their copies in `kernels/rvv_k3/` and pick the vector
register group from VLEN at run time (smallest group that holds the data:
both K3 cores charge by register group, not element count).
