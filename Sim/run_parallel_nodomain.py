#!/usr/bin/env python3
"""
Simple Parallel Data Collection (Without Domain Randomization)
Runs multiple Save_dataset.py instances in parallel

Usage:
CUDA_VISIBLE_DEVICES=0 python run_parallel_nodomain.py --workers 10 --episodes 500 \
    --base_dir /data/public/NAS/VLANeXt/dataset/New2
"""
import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers')
    parser.add_argument('--episodes', type=int, default=100, help='Episodes per worker')
    parser.add_argument('--base_dir', type=str,
                       default='/home/najo/NAS/VLA/Insertion_VLA_Sim2/Sim/collected_data_sim_6d_clean',
                       help='Base output directory')
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 Parallel Simulation Data Collection (No Domain Randomization)")
    print("=" * 80)
    print(f"Workers: {args.workers}")
    print(f"Episodes per worker: {args.episodes}")
    print(f"Total episodes: {args.workers * args.episodes}")
    print(f"Output: {args.base_dir}/worker_*")
    print(f"Phase tracking: ✅ Enabled")
    print("=" * 80)
    print()

    base_path = Path(args.base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    # Start workers
    processes = []
    for i in range(args.workers):
        worker_dir = base_path / f"worker_{i}"
        worker_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary Python script for this worker
        script_content = f"""
import sys
import os
import time

# Change to script directory
os.chdir('{Path(__file__).parent.absolute()}')
sys.path.insert(0, '{Path(__file__).parent.absolute()}')

# Import and modify Save_dataset
import Save_dataset

# Override settings
Save_dataset.SAVE_DIR = r'{worker_dir}'
Save_dataset.MAX_EPISODES = {args.episodes}

# Run
if __name__ == "__main__":
    print(f"[Worker {i}] Starting with MAX_EPISODES={args.episodes}, SAVE_DIR={worker_dir}")
    Save_dataset.main()
    print(f"[Worker {i}] Main completed, waiting for async file saves...")
    time.sleep(5)  # 비동기 저장 스레드 완료 대기
    print(f"[Worker {i}] All done!")
"""

        script_path = base_path / f"temp_worker_{i}.py"
        script_path.write_text(script_content)

        # Start process
        log_file = open(base_path / f"worker_{i}.log", 'w')
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).parent
        )
        processes.append((proc, log_file, script_path))
        print(f"[Worker {i}] Started (PID: {proc.pid})")
        time.sleep(1)  # Small delay between starts

    print()
    print("✅ All workers started!")
    print(f"Monitor logs: tail -f {base_path}/worker_*.log")
    print()
    print("Waiting for completion...")

    # Wait for all processes
    start_time = time.time()
    for i, (proc, log_file, script_path) in enumerate(processes):
        proc.wait()
        log_file.close()

        # 각 워커가 생성한 파일 개수 확인
        worker_dir = base_path / f"worker_{i}"
        h5_count = len(list(worker_dir.glob("*.h5"))) if worker_dir.exists() else 0
        print(f"[Worker {i}] Finished (exit code: {proc.returncode}, files: {h5_count})")

        script_path.unlink()  # Remove temp script

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print()
    print("=" * 80)
    print("✅ Collection Complete!")
    print("=" * 80)
    print(f"Time: {minutes}m {seconds}s")
    print()
    print("Merging data...")

    # Merge all worker directories
    final_dir = base_path / "collected_data_merged"
    final_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    successful_workers = 0
    failed_workers = 0

    for i in range(args.workers):
        worker_dir = base_path / f"worker_{i}"
        if worker_dir.exists():
            h5_files = list(worker_dir.glob("*.h5"))
            if h5_files:
                print(f"[Worker {i}] Moving {len(h5_files)} episodes...")
                for h5_file in h5_files:
                    new_name = f"worker{i}_{h5_file.name}"
                    shutil.move(str(h5_file), str(final_dir / new_name))
                    total_episodes += 1
                successful_workers += 1
                try:
                    worker_dir.rmdir()
                except:
                    pass  # Directory might not be empty
            else:
                print(f"[Worker {i}] ⚠️  No episodes collected")
                failed_workers += 1
        else:
            print(f"[Worker {i}] ⚠️  Directory not found")
            failed_workers += 1

    print()
    print("=" * 80)
    print("✅ Done!")
    print("=" * 80)
    print(f"Total episodes: {total_episodes}")
    print(f"Successful workers: {successful_workers}/{args.workers}")
    if failed_workers > 0:
        print(f"⚠️  Failed workers: {failed_workers}")
    print(f"Location: {final_dir}")
    print("=" * 80)
    print()
    print("📊 Phase Analysis:")
    print(f"    python analyze_phase_data.py {final_dir}")
    print()
    print("🎬 Export to MP4:")
    print(f"    for h5 in {final_dir}/*.h5; do")
    print(f'        python data_replay.py "$h5" --export --fps 30')
    print(f"    done")
    print("=" * 80)

if __name__ == "__main__":
    main()
