#!/usr/bin/env python3
"""squeue of my jobs with names/start (read-only)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import farm
rc, out, err = farm.ssh("squeue --me -h -o '%i|%j|%T|%N|%L|%S'")
print(out.decode(), err.decode()[-300:])
