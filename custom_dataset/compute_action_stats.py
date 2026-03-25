"""
Compute action normalization statistics (min/max) from HDF5 episode files.

Usage:
    python scripts/compute_action_stats.py --data_dir /path/to/collected_data_merged
"""

import argparse
import glob
import numpy as np
import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .h5 episode files")
    args = parser.parse_args()

    h5_files = sorted(glob.glob(f"{args.data_dir}/*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {args.data_dir}")
        return

    global_min = None
    global_max = None
    total_steps = 0

    for f_path in h5_files:
        try:
            with h5py.File(f_path, "r") as f:
                actions = f["action"][:].astype(np.float32)
                total_steps += actions.shape[0]
                fmin = actions.min(axis=0)
                fmax = actions.max(axis=0)
                if global_min is None:
                    global_min = fmin
                    global_max = fmax
                else:
                    global_min = np.minimum(global_min, fmin)
                    global_max = np.maximum(global_max, fmax)
        except Exception as e:
            print(f"[Warn] Skipping {f_path}: {e}")

    print(f"Episodes: {len(h5_files)}")
    print(f"Total steps: {total_steps:,}")
    print(f"Action dim: {global_min.shape[0]}")
    print()
    print(f"action_min_sim = {global_min.tolist()}")
    print(f"action_max_sim = {global_max.tolist()}")


if __name__ == "__main__":
    main()
