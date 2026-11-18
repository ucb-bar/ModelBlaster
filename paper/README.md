# Paper artifacts

- `plots/`: scripts that produce the paper's figures and tables (fast_p curves, KB statistics, optimization-state pie, RVV class plot, L40S summary).
- `../l3_farm/`: the batch driver (`farm.py`, `farm_sched.py`, `farm_defaults.py`, `config.json`), node helpers, `remote/` worker scripts, and the experiment `analysis/` and `plots/` scripts. API keys are read from a key file, not stored here.
- Data (per-problem result CSVs and the final knowledge-base snapshots) is in a separate bundle, `ModelBlaster_data_bundle.tar.gz`; see its README.
- Several scripts default to paths on the authors' machine (`/Users/krisd/Documents/ModelBlaster-clone/...`); pass the unpacked bundle paths as arguments or edit the defaults.
