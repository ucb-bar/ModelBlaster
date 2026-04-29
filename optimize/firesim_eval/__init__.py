"""FireSim re-rank step for the kernel optimize loop.

The base optimize loop in agents.pipeline.generate_kernels uses spike to
score candidate kernels — flat memory, fast turnaround, blind to cache
locality. This module adds an opt-in re-rank step that takes the top-K
spike survivors, builds them for the FireSim chipyard quad-rocket-saturn
hwconfig, runs them on the real RTL, and picks the kernel with the
lowest *firesim* per-op cycles. That promotes cache-locality wins which
look identical to pipeline wins on spike.

Public surface:

    from agents.optimize.firesim_eval import (
        FiresimEvaluator, FiresimEvalConfig, evaluate_top_k,
        memory_model_stanza,
    )
"""

from agents.optimize.firesim_eval.evaluator import (  # noqa: F401
    FiresimEvalConfig,
    FiresimEvalResult,
    FiresimEvaluator,
    evaluate_top_k,
)
from agents.optimize.firesim_eval.cache_aware_prompt import (  # noqa: F401
    QUAD_ROCKET_SATURN_MEMORY_MODEL,
    memory_model_stanza,
)
