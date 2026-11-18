#!/usr/bin/env python3
"""Collect the elapsed cycles from the definitive kernel for each RL baseline directory.

For every kernel folder under the given base directory, we track:
1. Baseline cycles from 0_init_annotated.cu
2. Best cycles for each iteration threshold (0-10), considering all iterations up to that threshold
3. For threshold n > 0, we consider files rl_iter_k_best.cu where k < n
4. For threshold 0, no iterations are considered (baseline only)
5. If no qualifying files exist for a threshold, we use the baseline result

We output a CSV with columns: Kernel ID, Baseline, Best_0, Best_1, ..., Best_10
where Best_n represents the best result considering all iterations < n.
"""
from __future__ import annotations
import argparse
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

def discover_kernels(base_path: Path) -> List[str]:
    return sorted([d.name for d in base_path.iterdir() if d.is_dir() and re.match(r"\d+_.*", d.name)])

def get_baseline_cycles(kernel_dir: Path) -> int | None:
    baseline_file = kernel_dir / "rl_ncu" / "0_init_annotated.cu"
    if not baseline_file.exists():
        return None
    text = baseline_file.read_text(errors="ignore")
    m = re.search(r"The Elapsed Cycles for this kernel is (\d+) cycle\.", text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None

def get_iter_files_up_to(kernel_dir: Path, max_iter: int) -> List[Path]:
    rl_ncu_dir = kernel_dir / "rl_ncu"
    if not rl_ncu_dir.is_dir():
        return []
    
    candidates = []
    for path in rl_ncu_dir.glob("rl_iter_*_best.cu"):
        m = re.search(r"rl_iter_(\d+)_best", path.name)
        if m and int(m.group(1)) < max_iter:
            candidates.append(path)
    return candidates

def extract_elapsed_cycles(file_path: Path) -> int | None:
    text = file_path.read_text(errors="ignore")
    # Try RL format first
    m = re.search(r"//\s*Elapsed Cycles:\s*(\d+)", text)
    if not m:
        # Try baseline format
        m = re.search(r"The Elapsed Cycles for this kernel is (\d+) cycle", text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None

def get_best_cycles_up_to_iter(kernel_dir: Path, max_iter: int, baseline_cycles: int | None) -> int | str:
    # For threshold 0, always return baseline (no iterations allowed)
    if max_iter == 0:
        return baseline_cycles if baseline_cycles is not None else "NO_DATA"
    
    iter_files = get_iter_files_up_to(kernel_dir, max_iter)
    if not iter_files:
        return baseline_cycles if baseline_cycles is not None else "NO_DATA"
    
    best_cycles = None
    for file in iter_files:
        cycles = extract_elapsed_cycles(file)
        if cycles is not None:
            best_cycles = min(cycles, best_cycles) if best_cycles is not None else cycles
    
    return best_cycles if best_cycles is not None else (baseline_cycles if baseline_cycles is not None else "NO_DATA")


def compute_cumulative_best_cycles(base_dir: Path, max_iter: int = 10, require_checker: bool = True) -> Tuple[List[str], List[List[Union[str, int, float]]]]:
    """Compute cumulative best cycles table for a base directory.

    Returns (header, rows):
      - header: ["Kernel ID", "Baseline", "Best_0", ..., "Best_{max_iter}"]
      - rows: per-kernel rows matching the header
    """
    kernels = discover_kernels(base_dir)
    header = ["Kernel ID", "Baseline"] + [f"Best_{i}" for i in range(max_iter + 1)]
    rows: List[List[Union[str, int, float]]] = []
    for kid in kernels:
        kernel_dir = base_dir / kid
        if require_checker:
            checker_file = kernel_dir / "rl_ncu" / "checker_passed.txt"
            if not checker_file.exists():
                continue
        baseline_cycles = get_baseline_cycles(kernel_dir)
        row: List[Union[str, int, float, str]] = [kid, baseline_cycles if baseline_cycles is not None else "NO_BASELINE"]
        for i in range(max_iter + 1):
            best_cycles = get_best_cycles_up_to_iter(kernel_dir, i, baseline_cycles)
            row.append(best_cycles)
        rows.append(row)
    return header, rows

def calculate_speedups(rows: List[List[Union[str, int, float]]]) -> List[List[Optional[float]]]:
    """Calculate speedups over baseline for each iteration.
    Returns a list of lists, where each sublist contains speedups for one iteration threshold.
    Only includes kernels that have valid baseline values."""
    speedups = []
    
    # Skip first two columns (Kernel ID and Baseline)
    for iter_idx in range(11):  # 0 to 10
        iter_speedups = []
        for row in rows:
            baseline = row[1]  # Baseline column
            iter_value = row[iter_idx + 2]  # +2 to skip Kernel ID and Baseline
            
            # Only include if both baseline and iter value are numbers
            if isinstance(baseline, (int, float)) and isinstance(iter_value, (int, float)):
                speedup = baseline / iter_value
                iter_speedups.append(speedup)
        
        speedups.append(iter_speedups)
    
    return speedups

def create_iqr_plot(speedups: List[List[float]], output_dir: Path):
    """Create IQR plots of speedups across iterations."""
    # Ensure plots directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate statistics for each iteration
    x = list(range(11))  # 0 to 10 iterations
    medians = []
    q25s = []
    q75s = []
    q10s = []
    q90s = []
    mins = []
    maxs = []
    
    for iter_speedups in speedups:
        if iter_speedups:  # Only if we have data
            medians.append(np.median(iter_speedups))
            q10s.append(np.percentile(iter_speedups, 10))
            q25s.append(np.percentile(iter_speedups, 25))
            q75s.append(np.percentile(iter_speedups, 75))
            q90s.append(np.percentile(iter_speedups, 90))
            mins.append(np.min(iter_speedups))
            maxs.append(np.max(iter_speedups))
        else:  # No data for this iteration
            medians.append(np.nan)
            q10s.append(np.nan)
            q25s.append(np.nan)
            q75s.append(np.nan)
            q90s.append(np.nan)
            mins.append(np.nan)
            maxs.append(np.nan)
    
    # Plot 1: Full IQR with min/max
    plt.figure(figsize=(10, 6))
    plt.plot(x, medians, 'b-', label='Median', linewidth=2)
    plt.fill_between(x, q25s, q75s, color='b', alpha=0.2, label='25-75 percentile')
    plt.plot(x, mins, 'b:', label='Min/Max', alpha=0.5)
    plt.plot(x, maxs, 'b:', alpha=0.5)
    
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('Iteration Threshold')
    plt.ylabel('Speedup over Baseline (log scale)')
    plt.title('Performance Improvement Across Num Trajectories')
    plt.legend()
    plt.yscale('log')
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.3)
    
    plt.savefig(output_dir / 'speedup_iqr_full.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Just median and quartiles
    plt.figure(figsize=(10, 6))
    plt.plot(x, medians, 'b-', label='Median', linewidth=2)
    plt.fill_between(x, q25s, q75s, color='b', alpha=0.2, label='25-75 percentile')
    
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('Num Trajectories')
    plt.ylabel('Speedup over Baseline (log scale)')
    plt.title('Performance Improvement Across Num Trajectories')
    plt.legend()
    plt.yscale('log')
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.3)
    
    # Format y-axis to use regular numbers instead of scientific notation
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter())
    plt.gca().yaxis.set_minor_formatter(plt.ScalarFormatter())
    
    plt.savefig(output_dir / 'speedup_iqr_central.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Median, quartiles, and deciles
    plt.figure(figsize=(5, 3))
    plt.plot(x, medians, 'b-', label='Median', linewidth=2)
    plt.fill_between(x, q25s, q75s, color='b', alpha=0.2, label='25-75 percentile')
    plt.plot(x, q10s, 'b:', label='10-90 percentile', alpha=0.5)
    plt.plot(x, q90s, 'b:', alpha=0.5)
    
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('Num Trajectories')
    plt.ylabel('Speedup over KGEN Baseline (log scale)')
    plt.title('Performance Improvement Across Num Trajectories')
    plt.legend()
    plt.yscale('log')
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.3)
    
    # Format y-axis to use regular numbers instead of scientific notation
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter())
    plt.gca().yaxis.set_minor_formatter(plt.ScalarFormatter())
    
    plt.savefig(output_dir / 'speedup_iqr_deciles.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Extract elapsed cycles tracking cumulative best results across iterations.")
    parser.add_argument("--base-dir", type=Path, action="append", required=True, help="Base directory to parse (repeatable)")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file or directory if multiple base dirs")
    parser.add_argument("--max-iter", type=int, default=10, help="Maximum iteration threshold (generates Best_0..Best_N)")
    parser.add_argument("--no-checker", action="store_true", help="Do not require rl_ncu/checker_passed.txt")
    parser.add_argument("--no-plots", action="store_true", help="Skip IQR plots")
    args = parser.parse_args()

    multi = len(args.base_dir) > 1
    out_path: Path = args.output
    if multi:
        # If multiple base dirs, treat output as a directory
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    for bdir in args.base_dir:
        header, rows = compute_cumulative_best_cycles(bdir, max_iter=args.max_iter, require_checker=not args.no_checker)
        if multi:
            out_file = out_path / f"{bdir.name}_cumulative_best_cycles.csv"
        else:
            out_file = out_path
        with out_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {out_file}")

        if not args.no_plots:
            # Calculate speedups and create plot (single base dir view)
            speedups = calculate_speedups(rows)
            plot_dir = Path("plots") / bdir.name if multi else Path("plots")
            create_iqr_plot(speedups, plot_dir)
            print(f"Created speedup IQR plots in {plot_dir}")

if __name__ == "__main__":
    main() 