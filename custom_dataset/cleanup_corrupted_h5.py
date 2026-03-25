"""
Detect and remove corrupted HDF5 episode files.

Usage:
    # Dry-run (default): only list corrupted files
    python dataset/cleanup_corrupted_h5.py --data_dir /path/to/merged

    # Actually delete corrupted files
    python dataset/cleanup_corrupted_h5.py --data_dir /path/to/merged --delete
"""

import argparse
import glob
import os
import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .h5 episode files")
    parser.add_argument("--delete", action="store_true", help="Actually delete corrupted files (default: dry-run)")
    args = parser.parse_args()

    h5_files = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))
    if not h5_files:
        print(f"No .h5 files found in {args.data_dir}")
        return

    print(f"Total h5 files: {len(h5_files)}")

    bad_files = []
    good_count = 0

    for f_path in h5_files:
        try:
            with h5py.File(f_path, "r") as f:
                _ = f["action"].shape
            good_count += 1
        except Exception as e:
            bad_files.append((f_path, str(e)))

    print(f"Good files: {good_count}")
    print(f"Corrupted files: {len(bad_files)}")

    if not bad_files:
        print("No corrupted files found.")
        return

    print()
    for f_path, err in bad_files:
        print(f"  [BAD] {os.path.basename(f_path)}: {err[:80]}")

    if args.delete:
        print()
        for f_path, _ in bad_files:
            os.remove(f_path)
            print(f"  [DELETED] {os.path.basename(f_path)}")
        print(f"\nDeleted {len(bad_files)} corrupted files. Remaining: {good_count}")
    else:
        print(f"\nDry-run mode. To delete, run again with --delete")


if __name__ == "__main__":
    main()
