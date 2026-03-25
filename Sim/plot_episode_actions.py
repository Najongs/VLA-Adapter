#!/usr/bin/env python
"""
Plot qpos, action, and ee_pose with confirmed units.

Units:
1. Action:  Linear=mm, Rotation=rad (Delta)
2. EE Pose: Linear=mm, Rotation=degree (Fixed by user)
3. Qpos:    All=degree

Usage:
python plot_episode_actions.py \
    --episode /home/najo/NAS/VLA/Insertion_VLA_Sim2/Sim/validation_result/validation_20260202_213939.h5 \
    --output_dir ./action_plots

python plot_episode_actions.py \
    --episode /home/najo/NAS/VLA/Insertion_VLA_Sim2/Sim/validation_results/validation_20260202_193413.h5 \
    --output_dir ./action_plots
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def find_episode_file(episode_input: str, base_dir: str = "/home/najo/NAS/VLA/dataset/New_dataset/collected_data"):
    """Find episode file in dataset directory."""
    episode_path = Path(episode_input)

    if episode_path.exists():
        return str(episode_path)

    episode_name = episode_path.name
    print(f"Searching for '{episode_name}' in {base_dir}...")

    search_pattern = f"{base_dir}/**/{episode_name}"
    matches = glob.glob(search_pattern, recursive=True)

    if not matches:
        raise FileNotFoundError(f"Episode file '{episode_name}' not found in {base_dir}")

    if len(matches) > 1:
        print(f"Warning: Found {len(matches)} matching files. Using first match: {matches[0]}")

    return matches[0]


def load_episode_data(h5_path: str):
    """Load qpos, action, and ee_pose data."""
    print(f"Loading episode from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        # 1. Load qpos
        try:
            qpos = f['observations']['qpos'][:]
        except KeyError:
            qpos = None

        # 2. Load ee_pose
        try:
            ee_pose = f['observations']['ee_pose'][:]
        except KeyError:
            ee_pose = None

        # 3. Load action
        try:
            if 'action' in f:
                action = f['action'][:]
            elif 'action' in f['observations']:
                action = f['observations']['action'][:]
            else:
                action = None
        except KeyError:
            action = None

        # 4. Load timestamps
        try:
            timestamps = f['timestamp'][:] if 'timestamp' in f else f['observations']['sensor']['timestamp'][:]
        except KeyError:
            timestamps = None

    return qpos, ee_pose, action, timestamps


def plot_data_6d(data, data_type, timestamps=None, output_dir=None, episode_name=""):
    """
    Plot 6D data with updated units (EE Pose -> Radian).
    """
    if data is None:
        print(f"Skipping plot for {data_type} (No data found)")
        return

    num_frames = len(data)
    data_dim = data.shape[1]
    x_values = np.arange(num_frames)
    x_label = 'Frame Index'

    # --- Unit & Label Definition Section ---
    
    # 1. Action: Linear=mm, Rot=rad
    if data_type == "action":
        dim_labels = [
            'dx (mm)', 'dy (mm)', 'dz (mm)', 
            'dRx (rad)', 'dRy (rad)', 'dRz (rad)'
        ]
        title_color = 'red'
        main_title = "Action (Delta: mm / rad)"

    # 2. EE Pose: Linear=mm, Rot=rad (Updated)
    elif data_type == "ee_pose":
        dim_labels = [
            'X (mm)', 'Y (mm)', 'Z (mm)', 
            'Rx (rad)', 'Ry (rad)', 'Rz (rad)'
        ]
        title_color = 'green'
        main_title = "EE Pose (Pos: mm / Rot: rad)"

    # 3. Qpos: All=degree
    elif data_type == "qpos":
        dim_labels = [f'Joint {i+1} (deg)' for i in range(6)]
        title_color = 'blue'
        main_title = "Joint Positions (Degree)"
    
    else:
        dim_labels = [f'Dim {i}' for i in range(6)]
        title_color = 'black'
        main_title = f"{data_type} Values"


    # --- Plotting ---
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'{main_title} - {episode_name}', fontsize=16, color=title_color, fontweight='bold')

    for i in range(min(data_dim, 6)):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        ax.plot(x_values, data[:, i], linewidth=1.5, color=title_color)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(dim_labels[i])
        ax.set_title(dim_labels[i])
        ax.grid(True, alpha=0.3)
        
        # Statistics
        mean_val = data[:, i].mean()
        min_val = data[:, i].min()
        max_val = data[:, i].max()
        
        # Added range info
        stats_text = f"Mean: {mean_val:.2f}\nRange: [{min_val:.2f}, {max_val:.2f}]"
        ax.legend([stats_text], loc='upper right', fontsize='small', framealpha=0.8)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"{data_type}_{episode_name}.png"
        save_path = output_path / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {data_type} plot to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_data_3d(data, data_type, output_dir=None, episode_name=""):
    if data is None:
        print(f"Skipping plot for {data_type} (No data found)")
        return

    num_frames = len(data)
    data_dim = data.shape[1]
    x_values = np.arange(num_frames)
    x_label = 'Frame Index'

    if data_type == "delta_ee_pos":
        dim_labels = ['dX (mm)', 'dY (mm)', 'dZ (mm)']
        title_color = 'purple'
        main_title = "Delta EE Position (mm)"
    else:
        dim_labels = [f'Dim {i}' for i in range(3)]
        title_color = 'black'
        main_title = f"{data_type} Values"

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'{main_title} - {episode_name}', fontsize=16, color=title_color, fontweight='bold')

    for i in range(min(data_dim, 3)):
        ax = axes[i]
        ax.plot(x_values, data[:, i], linewidth=1.5, color=title_color)
        ax.set_xlabel(x_label)
        ax.set_ylabel(dim_labels[i])
        ax.set_title(dim_labels[i])
        ax.grid(True, alpha=0.3)

        mean_val = data[:, i].mean()
        min_val = data[:, i].min()
        max_val = data[:, i].max()
        stats_text = f"Mean: {mean_val:.2f}\nRange: [{min_val:.2f}, {max_val:.2f}]"
        ax.legend([stats_text], loc='upper right', fontsize='small', framealpha=0.8)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"{data_type}_{episode_name}.png"
        save_path = output_path / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {data_type} plot to: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot with correct units")
    parser.add_argument("--episode", type=str, help="Path/filename of HDF5")
    parser.add_argument("--output_dir", type=str, default="./action_plots", help="Output directory")
    parser.add_argument("--base_dir", type=str, default="/home/najo/NAS/VLA/dataset/New_dataset/collected_data")

    args = parser.parse_args()

    try:
        episode_path = find_episode_file(args.episode, args.base_dir)
        episode_name = Path(episode_path).stem

        qpos, ee_pose, action, timestamps = load_episode_data(episode_path)

        # Generate plots
        plot_data_6d(qpos, "qpos", timestamps, args.output_dir, episode_name)
        plot_data_6d(ee_pose, "ee_pose", timestamps, args.output_dir, episode_name)
        plot_data_6d(action, "action", timestamps, args.output_dir, episode_name)
        if ee_pose is not None and ee_pose.shape[1] >= 6:
            delta_ee_pose = np.diff(ee_pose[:, :6], axis=0, prepend=ee_pose[:1, :6])
            plot_data_6d(delta_ee_pose, "action", timestamps, args.output_dir, f"{episode_name}_delta_ee_pose")

        print("\nAll plots generated successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
