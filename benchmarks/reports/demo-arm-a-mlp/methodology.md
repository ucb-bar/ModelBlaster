# Methodology: demo-arm-a-mlp

_Fill this in before sharing — without prose, numbers alone don't tell
the reader WHY you tried something or how to reproduce it._

## Approach
What did you try? (e.g. "Switch Arm B-bedrock from beam=2 exp=3 iter=2
to beam=1 exp=4 iter=1 to spend more on diversity per parent and less
on iteration depth.")

## Hypothesis
What did you expect would work, and why? (e.g. "Wider expansions per
parent should land a better kernel earlier without paying for extra
iterations on already-converged ops.")

## Knobs changed
| Knob | Default | This run |
|---|---|---|
| arm | A | (e.g. B-bedrock) |
| LLM model | — | (e.g. claude-sonnet-4-5) |
| beam | 2 | |
| expansions | 3 | |
| iterations | 2 | |
| FIRESIM_EVAL | 0 | |
| max_usd | none | |

## Result interpretation
What did the numbers show? Which kernels improved, by how much, and
why? Reference specific cells in this report:

  cells covered (first 5): 

See `report.md` for the cost + cycle table; `kernels/` for the actual
generated C code; `per-cell/<cell>/cycles_per_op.json` for per-op cycle
breakdowns.

## Reproducing this report
```bash
git checkout 7ea176a
# ... your commands here ...
mb-cost export --full demo-arm-a-mlp
```

## Next steps
What would you try next based on these results?
