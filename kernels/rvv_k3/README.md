# kernels/rvv_k3

Tuned kernels for the SpacemiT K3 X100 cores (CPUs 0-7, VLEN=256), backends
`rvv_k3` and `ime_k3_x100` (`-march=rv64gcv_zvl256b_zfh_zvfh`).

File naming: `rvv_k3_<op>_<algo>.c`. `rvv_k3` probes this directory first, and
`ime_k3_x100` probes it after `kernels/ime_k3_x100/`; any op without a file
here falls back to the shared `kernels/rvv/` kernels (backend alias `rvv`).
The K1 backends (`rvv_x60`, `ime_x60`) never read this directory, so nothing
here changes K1 code.

The simple-layer kernels here (k3-work TASK-017) have the same text as their
copies in `kernels/rvv_k3_a100/`: they pick the vector register group from
VLEN at run time. Both K3 cores charge a vector instruction by its register
group (LMUL), not by its element count, and `vrgather.vv` grows with LMUL
squared, so these kernels use the smallest group that holds the data. Every
one is bit-exact with the `kernels/rvv/` kernel it replaces.
