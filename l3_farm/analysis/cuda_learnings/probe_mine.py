#!/usr/bin/env python3
"""farm.probe of my nodes (read-only)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import farm
for j in map(int, sys.argv[1:]):
    print(j, json.dumps(farm.probe(j)))
