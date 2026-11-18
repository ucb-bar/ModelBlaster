"""Fixed inputs for the flags-off golden test (no new kwargs: valid for the pristine and the patched function)."""
def cases(p):
    sid = next(iter(p.slots))
    base = dict(source="// src\n__global__ void k(){}", technique="vectorize_loads",
                technique_desc="Use float4 loads.", state="memory_bound", state_numbers={"dram": 0.4},
                slot_us=12.5, partition_us=100.0, neighbours=[("other", "void other(float*)")],
                examples=[("ex", "// example kernel")], adapt=("[1]", "// adapted"), arch="89",
                knowledge_text="- (+3/-0) [pitfall] use __syncthreads", already_tried="- tiling: 2 tries (regressed 2)",
                best_note="Best verified so far for this slot: 10.0 us.", profile_text="PROFILE", runtime_text="RT")
    empty = dict(base, adapt=None, examples=[], neighbours=[], knowledge_text="", already_tried="", best_note="",
                 profile_text="", runtime_text="")
    return {"full": (sid, base), "empty": (sid, empty)}
