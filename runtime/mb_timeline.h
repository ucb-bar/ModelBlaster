/* mb_timeline.h -- opt-in per-frame timeline capture for the xpurt runtime (build with MB_TIMELINE=1).
 *
 * What it records, all in mcycle (core cycles, per hart):
 *   - per schedule entry (mb_tl_ent[entry]): claim (worker picks the entry), deps (all k_sem_takes
 *     returned), gate (schedule start-time gate passed), start / end (immediately around the kernel
 *     call), give (before the first k_sem_give), post (after the last), plus the hart and worker index.
 *   - per hart, an append-only event log (mb_tl_ev[hart][]): points logged by kernels or device drivers with
 *     MB_TL_EV(tag, aux), each tagged with the entry running on that hart. Tag values are defined by the code
 *     that logs them (0 is reserved); this header defines none.
 *   - frame start / done (mb_tl_hdr), taken at the same two points as the harness's wall_start_<net> and
 *     wall_cycles_<net> (mtime), so the frame time is the quantity the fps tables report.
 *   - cross-hart alignment (mb_tl_hdr.sync_*): after a shared arrival barrier, MB_TL_SYNC_ROUNDS NTP-style
 *     ping-pongs between worker 0 and every other worker, interrupts masked. For a round with master stamps
 *     t0 (ping sent), t2 (pong seen) and peer stamp t1: offset(peer - master) = t1 - (t0 + t2) / 2,
 *     |error| <= (t2 - t0) / 2. The host keeps the minimum-RTT round.
 *   - per hart (mb_tl_hs[hart]): (mcycle, mtime) pairs after the barrier and at worker exit, so the host can
 *     check that mcycle kept counting at the mtime rate (clock_hz / mtime_hz cycles per tick) while the hart slept.
 *
 * Nothing is printed: the statics are meant to be read from target memory after the run (e.g. over JTAG).
 * Cost: one rdcycle + 2-4 stores per stamp (the event macro also reads mhartid and bumps a counter).
 * Storage sits in .noinit (after .bss), so the existing data/bss layout does not move; main() zeroes it.
 *
 * Kernel translation units only use MB_TL_EV / MB_TL_SET_AUX (macros: no out-of-line helper, so nothing here
 * can be auto-vectorized into a TU built with a vector ISA that runs on a hart without a vector unit).
 * The generated main.c defines MB_TL_DEFINE_STORAGE and MB_TL_N_ENTRIES before including this file. */
#ifndef MB_TIMELINE_H
#define MB_TIMELINE_H
#include <stdint.h>

#define MB_TL_MAGIC   0x314C544Du   /* "MTL1" */
#define MB_TL_VERSION 1u
#ifndef MB_TL_MAX_HARTS
#define MB_TL_MAX_HARTS 2
#endif
#ifndef MB_TL_MAX_EV
#define MB_TL_MAX_EV 256            /* per hart, 16 B each */
#endif
#ifndef MB_TL_SYNC_ROUNDS
#define MB_TL_SYNC_ROUNDS 8
#endif

typedef struct {
    uint64_t cyc;
    uint16_t tag;
    int16_t  entry;     /* entry running on this hart when the event fired, -1 outside dispatches */
    uint32_t aux;
} mb_tl_ev_t;

typedef struct {
    uint64_t claim, deps, gate, start, end, give, post;
    uint16_t hart, worker;
    uint32_t flags;     /* bit0 ran, bit1 zero-cost entry (no kernel) */
} __attribute__((aligned(64))) mb_tl_ent_t;

typedef struct {
    uint32_t n_ev;
    int32_t  cur_entry;
    uint64_t sync_cyc0, sync_mtime, sync_cyc1;      /* after the barrier: mcycle, mtime, mcycle */
    uint64_t exit_cyc0, exit_mtime, exit_cyc1;      /* worker exit */
} __attribute__((aligned(64))) mb_tl_hart_t;

typedef struct {
    uint32_t magic, version;
    uint32_t n_entries, max_ev, max_harts, sync_rounds;
    uint32_t n_workers, sync_ok;                    /* sync_ok: 1 ping-pong ran, 0 skipped (1 worker / shared hart) */
    uint64_t frame_start_cyc, frame_done_cyc;       /* in the stamping hart's mcycle */
    int32_t  frame_start_hart, frame_done_hart;
    uint64_t frame_start_mtime, frame_done_mtime;
    uint64_t sync_t0[MB_TL_MAX_HARTS][MB_TL_SYNC_ROUNDS];   /* [peer worker][round], master clock */
    uint64_t sync_t1[MB_TL_MAX_HARTS][MB_TL_SYNC_ROUNDS];   /* peer clock */
    uint64_t sync_t2[MB_TL_MAX_HARTS][MB_TL_SYNC_ROUNDS];   /* master clock */
    int32_t  sync_master_hart, _pad;
    int32_t  sync_peer_hart[MB_TL_MAX_HARTS];               /* [peer worker] -> its hart */
} __attribute__((aligned(64))) mb_tl_hdr_t;

extern mb_tl_hart_t mb_tl_hs[MB_TL_MAX_HARTS];
extern mb_tl_ev_t mb_tl_ev[MB_TL_MAX_HARTS][MB_TL_MAX_EV];

/* CSR reads with Zicsr enabled locally: a kernel compiled with __attribute__((target("arch=rv64imac")))
 * loses zicsr (GCC 14), and the stamps must still assemble there. 0xC00 = cycle, 0xF14 = mhartid. */
#define MB_TL_CSRR(csr_) ({ unsigned long _v_;                                        \
        __asm__ volatile(".option push\n\t.option arch, +zicsr\n\tcsrr %0, " #csr_ "\n\t.option pop" \
                         : "=r"(_v_)); _v_; })
#define MB_TL_RDCYCLE() ((uint64_t)MB_TL_CSRR(0xc00))
#define MB_TL_HARTID()  MB_TL_CSRR(0xf14)

/* Append one event to this hart's log. */
#define MB_TL_EV(tag_, aux_) do {                                                  \
        const uint64_t _t_ = MB_TL_RDCYCLE();                                      \
        const unsigned long _hh_ = MB_TL_HARTID();                                 \
        if (_hh_ < MB_TL_MAX_HARTS) {                                              \
            mb_tl_hart_t *_s_ = &mb_tl_hs[_hh_];                                   \
            const uint32_t _n_ = _s_->n_ev;                                        \
            if (_n_ < MB_TL_MAX_EV) {                                              \
                mb_tl_ev_t *_e_ = &mb_tl_ev[_hh_][_n_];                            \
                _e_->cyc = _t_; _e_->tag = (uint16_t)(tag_);                       \
                _e_->entry = (int16_t)_s_->cur_entry; _e_->aux = (uint32_t)(aux_); \
            }                                                                      \
            _s_->n_ev = _n_ + 1;                                                   \
        }                                                                          \
    } while (0)

/* Set the aux word of this hart's last event (e.g. a device cycle counter read after the done stamp). */
#define MB_TL_SET_AUX(aux_) do {                                                   \
        const unsigned long _hh_ = MB_TL_HARTID();                                 \
        if (_hh_ < MB_TL_MAX_HARTS) {                                              \
            const uint32_t _n_ = mb_tl_hs[_hh_].n_ev;                              \
            if (_n_ >= 1 && _n_ <= MB_TL_MAX_EV)                                   \
                mb_tl_ev[_hh_][_n_ - 1].aux = (uint32_t)(aux_);                    \
        }                                                                          \
    } while (0)

#ifdef MB_TL_DEFINE_STORAGE
/* ------------------------------------------------------------------------------------------------
 * Storage and the harness side; only the generated main.c (built without a vector ISA) includes this. */
#include <zephyr/kernel.h>
#define MB_TL_NOINIT __attribute__((section(".noinit.mb_tl"), aligned(64)))
mb_tl_hdr_t  mb_tl_hdr MB_TL_NOINIT;
mb_tl_ent_t  mb_tl_ent[MB_TL_N_ENTRIES] MB_TL_NOINIT;
mb_tl_hart_t mb_tl_hs[MB_TL_MAX_HARTS] MB_TL_NOINIT;
mb_tl_ev_t   mb_tl_ev[MB_TL_MAX_HARTS][MB_TL_MAX_EV] MB_TL_NOINIT;
/* ping / pong / arrival counter, one line each */
static struct { volatile uint64_t v; uint8_t pad[56]; } mb_tl_ping MB_TL_NOINIT, mb_tl_pong MB_TL_NOINIT,
    mb_tl_arrive MB_TL_NOINIT;

/* Zero a region with 8-byte volatile stores (no memset: keeps the code scalar and predictable). */
static inline void mb_tl_zero(volatile void *p, unsigned long bytes)
{
    volatile uint64_t *q = (volatile uint64_t *)p;
    for (unsigned long i = 0; i < bytes / 8; i++) q[i] = 0;
}

/* main(), before any worker exists. */
static inline void mb_tl_init(int n_workers)
{
    static int32_t runs;                /* re-arms so far (a multi-run main re-arms once per run) */
    mb_tl_zero(&mb_tl_hdr, sizeof mb_tl_hdr);
    mb_tl_zero(mb_tl_ent, sizeof mb_tl_ent);
    mb_tl_zero(mb_tl_hs, sizeof mb_tl_hs);
    mb_tl_zero(&mb_tl_ping, sizeof mb_tl_ping);
    mb_tl_zero(&mb_tl_pong, sizeof mb_tl_pong);
    mb_tl_zero(&mb_tl_arrive, sizeof mb_tl_arrive);
    for (int h = 0; h < MB_TL_MAX_HARTS; h++) mb_tl_hs[h].cur_entry = -1;
    mb_tl_hdr.n_entries = MB_TL_N_ENTRIES;
    mb_tl_hdr.max_ev = MB_TL_MAX_EV;
    mb_tl_hdr.max_harts = MB_TL_MAX_HARTS;
    mb_tl_hdr.sync_rounds = MB_TL_SYNC_ROUNDS;
    mb_tl_hdr.n_workers = (uint32_t)n_workers;
    mb_tl_hdr.frame_start_hart = mb_tl_hdr.frame_done_hart = -1;
    mb_tl_hdr.sync_master_hart = -1;
    mb_tl_hdr._pad = runs++;            /* run index of this capture (0 = first run of the boot) */
    for (int h = 0; h < MB_TL_MAX_HARTS; h++) mb_tl_hdr.sync_peer_hart[h] = -1;
    __asm__ volatile("fence rw, rw" ::: "memory");
    mb_tl_hdr.magic = MB_TL_MAGIC;      /* last: a readback that sees the magic sees a consistent header */
    mb_tl_hdr.version = MB_TL_VERSION;
}

/* Every worker, once, before its first entry. `distinct` = every worker has its own hart.
 * Pre-touches this hart's event log (so the first stamps of the frame are cache hits), meets the other
 * workers at an arrival barrier, then (distinct harts only) runs the ping-pong with worker 0. */
static inline void mb_tl_worker_sync(int worker, int hart, int n_workers, int distinct)
{
    if (hart >= 0 && hart < MB_TL_MAX_HARTS) {
        mb_tl_zero(mb_tl_ev[hart], sizeof mb_tl_ev[hart]);
        mb_tl_hs[hart].n_ev = 0;
        mb_tl_hs[hart].cur_entry = -1;
    }
    __atomic_fetch_add(&mb_tl_arrive.v, 1, __ATOMIC_ACQ_REL);
    for (unsigned long spin = 0; __atomic_load_n(&mb_tl_arrive.v, __ATOMIC_ACQUIRE) < (uint64_t)n_workers; spin++)
        if ((spin & 1023) == 1023) k_yield();
    const int do_sync = distinct && n_workers > 1 && n_workers <= MB_TL_MAX_HARTS &&
                        hart >= 0 && hart < MB_TL_MAX_HARTS;
    if (do_sync) {
        unsigned int key = arch_irq_lock();
        if (worker == 0) {
            mb_tl_hdr.sync_master_hart = hart;
            for (int p = 1; p < n_workers; p++)
                for (int r = 0; r < MB_TL_SYNC_ROUNDS; r++) {
                    const uint64_t seq = ((uint64_t)p << 16) | (uint64_t)(r + 1);
                    const uint64_t t0 = MB_TL_RDCYCLE();
                    __atomic_store_n(&mb_tl_ping.v, seq, __ATOMIC_RELEASE);
                    while (__atomic_load_n(&mb_tl_pong.v, __ATOMIC_ACQUIRE) != seq) { }
                    const uint64_t t2 = MB_TL_RDCYCLE();
                    mb_tl_hdr.sync_t0[p][r] = t0;   /* the peer stored t1 before its pong */
                    mb_tl_hdr.sync_t2[p][r] = t2;
                }
            mb_tl_hdr.sync_ok = 1;
        } else {
            mb_tl_hdr.sync_peer_hart[worker] = hart;
            for (int r = 0; r < MB_TL_SYNC_ROUNDS; r++) {
                const uint64_t seq = ((uint64_t)worker << 16) | (uint64_t)(r + 1);
                while (__atomic_load_n(&mb_tl_ping.v, __ATOMIC_ACQUIRE) != seq) { }
                const uint64_t t1 = MB_TL_RDCYCLE();
                mb_tl_hdr.sync_t1[worker][r] = t1;
                __atomic_store_n(&mb_tl_pong.v, seq, __ATOMIC_RELEASE);
            }
        }
        arch_irq_unlock(key);
    }
    if (hart >= 0 && hart < MB_TL_MAX_HARTS) {
        mb_tl_hs[hart].sync_cyc0 = MB_TL_RDCYCLE();
        mb_tl_hs[hart].sync_mtime = (uint64_t)k_cycle_get_64();
        mb_tl_hs[hart].sync_cyc1 = MB_TL_RDCYCLE();
    }
}

static inline void mb_tl_worker_exit(int hart)
{
    if (hart >= 0 && hart < MB_TL_MAX_HARTS) {
        mb_tl_hs[hart].exit_cyc0 = MB_TL_RDCYCLE();
        mb_tl_hs[hart].exit_mtime = (uint64_t)k_cycle_get_64();
        mb_tl_hs[hart].exit_cyc1 = MB_TL_RDCYCLE();
    }
}

/* Entry stamps (worker side; one writer per entry slot, 64-byte slots). */
#define MB_TL_ENT(i_, field_) (mb_tl_ent[(i_)].field_ = MB_TL_RDCYCLE())
#define MB_TL_ENT_CLAIM(i_, hart_, worker_) do {                                    \
        mb_tl_ent[(i_)].claim = MB_TL_RDCYCLE();                                     \
        mb_tl_ent[(i_)].hart = (uint16_t)(hart_);                                    \
        mb_tl_ent[(i_)].worker = (uint16_t)(worker_);                                \
    } while (0)
#define MB_TL_CUR(hart_, i_) do {                                                    \
        if ((hart_) >= 0 && (hart_) < MB_TL_MAX_HARTS) mb_tl_hs[(hart_)].cur_entry = (i_); \
    } while (0)
#endif /* MB_TL_DEFINE_STORAGE */
#endif /* MB_TIMELINE_H */
