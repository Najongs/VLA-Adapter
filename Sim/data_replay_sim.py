import h5py
import cv2
import numpy as np
import os
import argparse

# ==============================================================
# Dataset Viewer with MP4 Export
# Supports real-time viewing and video export with metadata overlay
# ==============================================================

# Phase mapping for display
PHASE_NAMES = {
    1: "Align",
    2: "Insert",
    3: "Hold",
    -1: "Unknown"
}

PHASE_COLORS = {
    1: (0, 255, 255),    # Yellow
    2: (0, 165, 255),    # Orange
    3: (0, 255, 0),      # Green
    -1: (128, 128, 128)  # Gray
}

def print_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"📁 [Group] {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"   💾 [Dataset] {name} | Shape: {obj.shape} | Type: {obj.dtype}")

def create_info_panel(current_step, total_steps, cur_q, cur_act, cur_ee, cur_phase, cur_sensor, width, is_playing=True):
    # 센서 값 표시를 위해 높이를 200으로 약간 확장
    info_panel = np.zeros((200, width, 3), dtype=np.uint8)

    green = (0, 255, 0)
    white = (255, 255, 255)
    yellow = (0, 255, 255)
    red = (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    phase_name = PHASE_NAMES.get(cur_phase, "Unknown")
    phase_color = PHASE_COLORS.get(cur_phase, (128, 128, 128))

    # Line 1: Step and Phase
    status = 'PLAY' if is_playing else 'PAUSE'
    cv2.putText(info_panel, f"Step: {current_step}/{total_steps} ({status})", (20, 25), font, 0.7, green, 2)
    cv2.putText(info_panel, f"Phase: {phase_name} ({cur_phase})", (width - 300, 25), font, 0.7, phase_color, 2)

    # Line 2: Joint positions (qpos)
    q_str = "Qpos (deg): " + " ".join([f"{x: .2f}" for x in cur_q])
    cv2.putText(info_panel, q_str, (20, 55), font, 0.5, white, 1)

    # Line 3: Action (Delta)
    a_str = "Delta Act (mm/rad): " + " ".join([f"{x: .4f}" for x in cur_act])
    cv2.putText(info_panel, a_str, (20, 80), font, 0.5, yellow, 1)

    # Line 4: EE Pose
    ee_str_pos = f"EE Abs (mm): X={cur_ee[0]:.2f} Y={cur_ee[1]:.2f} Z={cur_ee[2]:.2f}"
    ee_str_rot = f"EE Rot (rad): R={cur_ee[3]:.4f} P={cur_ee[4]:.4f} Y={cur_ee[5]:.4f}"
    cv2.putText(info_panel, ee_str_pos, (20, 110), font, 0.5, white, 1)
    cv2.putText(info_panel, ee_str_rot, (20, 135), font, 0.5, white, 1)

    # Line 5: [NEW] Sensor Distance
    if cur_sensor > 0:
        sensor_text = f"Ray-Cast Distance: {cur_sensor:.2f} mm"
        # 50mm(SENSOR_THRESHOLD) 이내면 빨간색으로 경고
        s_color = red if cur_sensor < 50.0 else green
    else:
        sensor_text = "Ray-Cast Distance: Out of Range / No Detection"
        s_color = (128, 128, 128)
    cv2.putText(info_panel, sensor_text, (20, 165), font, 0.6, s_color, 2)

    # Line 6: Controls
    cv2.putText(info_panel, "Controls: SPACE=Pause | A=Prev | D=Next | Q=Quit | S=Save MP4", (20, 190), font, 0.4, (200, 200, 200), 1)

    return info_panel
def save_to_mp4(hdf5_path, output_path=None, fps=30):
    if output_path is None:
        output_path = hdf5_path.replace('.h5', '.mp4')

    print(f"\n🎬 Exporting to MP4: {output_path}")

    with h5py.File(hdf5_path, 'r') as f:
        qpos_data = f['observations/qpos'][:]
        action_data = f['action'][:]
        ee_data = f['observations/ee_pose'][:]
        # Sensor data 로드 (없을 경우 -1로 채움)
        sensor_data = f['observations/sensor_dist'][:] if 'observations/sensor_dist' in f else np.full(len(qpos_data), -1.0)
        phase_data = f['phase'][:] if 'phase' in f else np.full(len(qpos_data), -1, dtype=np.int32)

        img_grp = f['observations/images']
        cam_keys = sorted(list(img_grp.keys()))
        total_steps = len(qpos_data)

        # Decode first frame to get dimensions
        frames = [cv2.imdecode(img_grp[k][0], cv2.IMREAD_COLOR) for k in cam_keys]
        combined_img = np.hstack(frames)
        h, w, _ = combined_img.shape

        info_panel = create_info_panel(0, total_steps, qpos_data[0], action_data[0], ee_data[0], phase_data[0], sensor_data[0], w)
        final_h, final_w = h + info_panel.shape[0], w

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (final_w, final_h))

        for step in range(total_steps):
            frames = [cv2.imdecode(img_grp[k][step], cv2.IMREAD_COLOR) for k in cam_keys]
            combined_img = np.hstack(frames)
            info_panel = create_info_panel(step, total_steps, qpos_data[step], action_data[step], ee_data[step], phase_data[step], sensor_data[step], w, is_playing=False)
            final_frame = np.vstack([combined_img, info_panel])
            out.write(final_frame)
            if (step + 1) % 30 == 0: print(f"  Progress: {(step+1)/total_steps*100:.1f}%", end='\r')

        out.release()
        print(f"\n✅ Video saved: {output_path}")

def view_interactive(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        qpos_data, action_data, ee_data = f['observations/qpos'][:], f['action'][:], f['observations/ee_pose'][:]
        sensor_data = f['observations/sensor_dist'][:] if 'observations/sensor_dist' in f else np.full(len(qpos_data), -1.0)
        phase_data = f['phase'][:] if 'phase' in f else np.full(len(qpos_data), -1, dtype=np.int32)
        
        img_grp = f['observations/images']
        cam_keys = sorted(list(img_grp.keys()))
        total_steps = len(qpos_data)

        cv2.namedWindow("Dataset Viewer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Dataset Viewer", 1200, 800)
        cv2.createTrackbar("Step", "Dataset Viewer", 0, total_steps - 1, lambda x: None)

        is_playing, current_step = True, 0
        while True:
            if is_playing:
                current_step = (current_step + 1) % total_steps
                cv2.setTrackbarPos("Step", "Dataset Viewer", current_step)
            else:
                current_step = cv2.getTrackbarPos("Step", "Dataset Viewer")

            frames = [cv2.imdecode(img_grp[k][current_step], cv2.IMREAD_COLOR) for k in cam_keys]
            combined_img = np.hstack(frames)
            info_panel = create_info_panel(current_step, total_steps, qpos_data[current_step], action_data[current_step], ee_data[current_step], phase_data[current_step], sensor_data[current_step], combined_img.shape[1], is_playing)
            cv2.imshow("Dataset Viewer", np.vstack([combined_img, info_panel]))

            key = cv2.waitKey(33) & 0xFF
            if key == ord('q'): break
            elif key == 32: is_playing = not is_playing
            elif key == ord('a'): is_playing = False; cv2.setTrackbarPos("Step", "Dataset Viewer", max(0, current_step - 1))
            elif key == ord('d'): is_playing = False; cv2.setTrackbarPos("Step", "Dataset Viewer", min(total_steps - 1, current_step + 1))
            elif key == ord('s'): save_to_mp4(hdf5_path)

    cv2.destroyAllWindows()

def check_display_available():
    """Check if display is available for interactive viewing."""
    import os
    # Check DISPLAY environment variable
    if 'DISPLAY' not in os.environ or not os.environ['DISPLAY']:
        return False
    # Don't actually try to create window - it might hang
    # Just check if we're in a headless environment
    return False  # Default to headless/export mode for safety

def main():
    parser = argparse.ArgumentParser(
        description="HDF5 Dataset Viewer and MP4 Exporter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive viewer (requires display)
  python data_replay.py /path/to/episode.h5

  # Export to MP4 (headless mode)
  python data_replay.py /path/to/episode.h5 --export

  # Export with custom output path and fps
  python data_replay.py /path/to/episode.h5 --export --output /path/to/output.mp4 --fps 60
        """
    )

    parser.add_argument(
        "hdf5_path",
        type=str,
        help="Path to HDF5 episode file"
    )

    parser.add_argument(
        "--export",
        action="store_true",
        help="Export to MP4 instead of interactive viewing"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output MP4 path (default: same as input with .mp4 extension)"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Video frame rate for export (default: 30)"
    )

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.hdf5_path):
        print(f"❌ File not found: {args.hdf5_path}")
        return

    # Export or view
    if args.export:
        # Explicit export mode
        save_to_mp4(args.hdf5_path, args.output, args.fps)
    else:
        # Check if display is available
        if not check_display_available():
            print("⚠️  No display detected - running in headless mode")
            print("🎬 Automatically switching to export mode...")
            save_to_mp4(args.hdf5_path, args.output, args.fps)
        else:
            view_interactive(args.hdf5_path)

if __name__ == "__main__":
    main()
