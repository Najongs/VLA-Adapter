"""
python reproduce_episode.py collected_data_sim_clean/episode_20260202_193258.h5 --output_file collected_data_sim_clean/reproduced_episode.h5
"""


import os
os.environ['MUJOCO_GL'] = 'egl'

import mujoco
import numpy as np
import cv2
import h5py
import argparse
import pathlib
from tqdm import tqdm

# === Configuration (Must match Save_dataset.py) ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "meca_add.xml")
IMG_WIDTH = 640
IMG_HEIGHT = 480

class ReplayRecorder:
    def __init__(self, output_path):
        self.output_path = output_path
        self.buffer = []
        self.metadata = {}

    def set_metadata(self, metadata):
        self.metadata = dict(metadata or {})

    def add(self, frames, qpos, ee_pose, action, timestamp, phase, sensor_dist):
        self.buffer.append({
            "ts": timestamp,
            "imgs": frames,
            "q": qpos,
            "p": ee_pose,
            "act": action,
            "phase": phase,
            "sd": sensor_dist
        })

    def save(self):
        if not self.buffer:
            print("No data to save.")
            return

        print(f"Saving reproduced data to {self.output_path}...")
        try:
            with h5py.File(self.output_path, 'w') as f:
                obs = f.create_group("observations")
                img_grp = obs.create_group("images")
                if self.metadata:
                    meta_grp = f.create_group("metadata")

                # Combine list of dicts into arrays
                q_data = np.array([x['q'] for x in self.buffer], dtype=np.float32)
                p_data = np.array([x['p'] for x in self.buffer], dtype=np.float32)
                act_data = np.array([x['act'] for x in self.buffer], dtype=np.float32)
                ts_data = np.array([x['ts'] for x in self.buffer], dtype=np.float32)
                phase_data = np.array([x['phase'] for x in self.buffer], dtype=np.int32)
                sensor_data = np.array([x['sd'] for x in self.buffer], dtype=np.float32)

                obs.create_dataset("qpos", data=q_data, compression="gzip")
                obs.create_dataset("ee_pose", data=p_data, compression="gzip")
                obs.create_dataset("sensor_dist", data=sensor_data, compression="gzip")
                f.create_dataset("action", data=act_data, compression="gzip")
                f.create_dataset("timestamp", data=ts_data, compression="gzip")
                f.create_dataset("phase", data=phase_data, compression="gzip")

                if self.metadata:
                    for key, value in self.metadata.items():
                        value_arr = np.asarray(value)
                        if value_arr.shape == ():
                            meta_grp.create_dataset(key, data=value_arr)
                        else:
                            meta_grp.create_dataset(key, data=value_arr, compression="gzip")

                # Save images
                first_imgs = self.buffer[0]["imgs"]
                for cam_name in first_imgs.keys():
                    jpeg_list = []
                    for step in self.buffer:
                        img = step["imgs"][cam_name]
                        success, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        if success:
                            jpeg_list.append(buf.flatten())
                        else:
                            jpeg_list.append(np.zeros(1, dtype=np.uint8))

                    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                    dset = img_grp.create_dataset(cam_name, (len(jpeg_list),), dtype=dt)
                    for i, code in enumerate(jpeg_list):
                        dset[i] = code
            print("✅ Save complete.")

        except Exception as e:
            print(f"❌ Save Failed: {e}")

def get_ee_pose_6d_scaled(model, data, link6_id):
    """Get 6_link world pose (x, y, z, rx, ry, rz). Matching Save_dataset.py logic."""
    if link6_id >= 0:
        pos = data.xpos[link6_id].copy() * 1000
        mat = data.xmat[link6_id].reshape(3, 3)
        sy = np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2)
        if sy > 1e-6:
            r = np.arctan2(mat[2, 1], mat[2, 2])
            p = np.arctan2(-mat[2, 0], sy)
            y = np.arctan2(mat[1, 0], mat[0, 0])
        else:
            r = np.arctan2(-mat[1, 2], mat[1, 1])
            p = np.arctan2(-mat[2, 0], sy)
            y = 0.0
        return np.concatenate([pos, [r, p, y]])
    return np.zeros(6, dtype=np.float32)


def load_episode_metadata(h5_file):
    metadata = {}
    if 'metadata' not in h5_file:
        return metadata

    meta_group = h5_file['metadata']
    for key in meta_group.keys():
        metadata[key] = meta_group[key][()]

    return metadata


def apply_episode_metadata(model, data, metadata):
    if not metadata:
        print("No episode metadata found; using default phantom pose.")
        return

    phantom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "phantom_assembly")
    rotating_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rotating_assembly")

    if "phantom_offset" in metadata:
        model.body_pos[phantom_id] = np.asarray(metadata["phantom_offset"], dtype=np.float64)
        print(f"Applied phantom offset: {model.body_pos[phantom_id]}")

    if "phantom_quat" in metadata:
        quat = np.asarray(metadata["phantom_quat"], dtype=np.float64)
        model.body_quat[rotating_id] = quat
        print(f"Applied phantom quaternion: {quat}")

    mujoco.mj_forward(model, data)

def main():
    parser = argparse.ArgumentParser(description="Reproduce dataset from joint angles.")
    parser.add_argument("input_file", type=str, help="Path to the source HDF5 file")
    parser.add_argument("--output_file", type=str, default="reproduced_episode.h5", help="Path to save the reproduced HDF5 file")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        return

    print(f"🔄 Loading Model: {MODEL_PATH}")
    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Get IDs (matching Save_dataset.py)
    try:
        tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
        back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
        link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "6_Link")
        n_motors = model.nu
    except Exception as e:
        print(f"⚠️ Warning: Some IDs not found: {e}")
        link6_id = -1
        n_motors = 6 # Default fallback

    # Load source data
    print(f"📖 Reading source data from {args.input_file}...")
    with h5py.File(args.input_file, 'r') as f:
        # Load necessary data to reproduce
        # We drive the robot using qpos.
        source_qpos = f['observations/qpos'][:] # shape (N, n_motors)
        
        # We also carry over other metadata for comparison or completeness,
        # though strictly we recalculate EE pose and Sensor dist to verify simulation.
        # But to match the file structure, we might want to preserve action/phase/timestamp from source
        # OR recalculate what we can. 
        # The prompt says: "check if it is reproduced exactly... give the same joint angle... save as h5 including images".
        # So we should probably recalculate EE pose and sensor dist to PROVE it's the same.
        # Actions, timestamps, phases are not "simulated" in this replay (they are logic states), 
        # so we can just copy them to the new file for context, or leave them as placeholders.
        # Let's copy them to allow full file comparison.
        
        source_action = f['action'][:]
        source_timestamp = f['timestamp'][:]
        source_phase = f['phase'][:]
        source_metadata = load_episode_metadata(f)
        
        # Ensure lengths match
        n_frames = len(source_qpos)
        print(f"Found {n_frames} frames.")

    recorder = ReplayRecorder(args.output_file)
    recorder.set_metadata(source_metadata)
    
    # Important: The source qpos is in DEGREES (based on Save_dataset.py).
    # Mujoco expects RADIANS.
    
    print("🚀 Starting Reproduction...")
    
    # Reset simulation
    mujoco.mj_resetData(model, data)
    apply_episode_metadata(model, data, source_metadata)

    last_ee_pose = None # To calculate delta if we wanted to, but we copy action.

    for i in tqdm(range(n_frames), desc="Reproducing"):
        # 1. Set Joint Angles
        # source_qpos[i] is in degrees.
        qpos_deg = source_qpos[i]
        qpos_rad = np.deg2rad(qpos_deg)
        
        # Set state
        # Assuming the first n_motors joints correspond to the robot.
        if len(qpos_rad) <= len(data.qpos):
            data.qpos[:len(qpos_rad)] = qpos_rad
        else:
             print(f"Warning: Source qpos dim {len(qpos_rad)} > data.qpos {len(data.qpos)}")
        
        # Forward kinematics (compute positions/sensors)
        mujoco.mj_forward(model, data)
        
        # 2. Recalculate Observed Data
        # EE Pose
        current_ee_pose_mm = get_ee_pose_6d_scaled(model, data, link6_id)
        
        # Sensor Dist
        p_sensor = data.site_xpos[tip_id].copy()
        curr_tip = data.site_xpos[tip_id].copy()
        curr_back = data.site_xpos[back_id].copy()
        needle_dir = (curr_tip - curr_back) / (np.linalg.norm(curr_tip - curr_back) + 1e-10)
        dist_to_surface = mujoco.mj_ray(model, data, p_sensor, needle_dir, None, 1, link6_id, np.zeros(1, dtype=np.int32))
        current_sensor_dist = dist_to_surface * 1000.0 if dist_to_surface >= 0 else -1.0
        
        # 3. Render Images
        frames = {}
        for cam_name in ["side_camera", "tool_camera", "top_camera"]:
            renderer.update_scene(data, camera=cam_name)
            frames[cam_name] = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            
        # 4. Add to buffer
        # We use the RECALCULATED qpos (which is just the input), EE pose, and sensor dist.
        # We use the ORIGINAL action, timestamp, phase (as they are history/logic data).
        # Actually, qpos in 'add' expects DEGREES.
        
        recorder.add(
            frames=frames,
            qpos=qpos_deg, 
            ee_pose=current_ee_pose_mm,
            action=source_action[i], # Copy original action
            timestamp=source_timestamp[i], # Copy original timestamp
            phase=source_phase[i], # Copy original phase
            sensor_dist=current_sensor_dist # Recalculated sensor
        )

    # Save
    recorder.save()

if __name__ == "__main__":
    main()
