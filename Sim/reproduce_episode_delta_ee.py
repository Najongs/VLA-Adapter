import os
os.environ['MUJOCO_GL'] = 'egl'

import mujoco
import numpy as np
import cv2
import h5py
import argparse
from tqdm import tqdm

# === Configuration ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "meca_add.xml")
IMG_WIDTH = 640
IMG_HEIGHT = 480

class ReplayRecorder:
    def __init__(self, output_path):
        self.output_path = output_path
        self.buffer = []

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

        print(f"Saving Delta-EE-reproduced data to {self.output_path}...")
        try:
            with h5py.File(self.output_path, 'w') as f:
                obs = f.create_group("observations")
                img_grp = obs.create_group("images")

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

# === Math Helpers ===
def euler_to_rot_mat(r, p, y):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r), np.cos(r)]
    ])
    Ry = np.array([
        [np.cos(p), 0, np.sin(p)],
        [0, 1, 0],
        [-np.sin(p), 0, np.cos(p)]
    ])
    Rz = np.array([
        [np.cos(y), -np.sin(y), 0],
        [np.sin(y), np.cos(y), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

def get_ee_pose_6d_scaled(data, link6_id):
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

def solve_ik(model, data, target_pos_m, target_rot_mat, link_id, q_init, n_motors, max_iter=500):
    data.qpos[:n_motors] = q_init
    mujoco.mj_forward(model, data)
    
    pos_tol = 1e-6
    rot_tol = 1e-5
    
    for _ in range(max_iter):
        curr_pos = data.xpos[link_id]
        curr_mat = data.xmat[link_id].reshape(3, 3)
        err_pos = target_pos_m - curr_pos
        R_err = target_rot_mat @ curr_mat.T
        
        quat_err = np.zeros(4)
        mujoco.mju_mat2Quat(quat_err, R_err.flatten())
        if quat_err[0] >= 1.0:
            err_rot = np.zeros(3)
        else:
            theta = 2 * np.arccos(np.clip(quat_err[0], -1, 1))
            sin_half = np.sqrt(1 - quat_err[0]**2)
            if sin_half < 1e-6:
                err_rot = np.zeros(3)
            else:
                err_rot = theta * (quat_err[1:] / sin_half)

        if np.linalg.norm(err_pos) < pos_tol and np.linalg.norm(err_rot) < rot_tol:
            return data.qpos[:n_motors].copy(), True
            
        jac_pos = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jac_pos, jac_rot, link_id)
        J = np.vstack([jac_pos[:, :n_motors], jac_rot[:, :n_motors]])
        err = np.concatenate([err_pos, err_rot])
        lamb = 1e-2
        dq = J.T @ np.linalg.solve(J @ J.T + lamb**2 * np.eye(6), err)
        data.qpos[:n_motors] += dq
        mujoco.mj_forward(model, data)
        
    return data.qpos[:n_motors].copy(), False

def main():
    parser = argparse.ArgumentParser(description="Reproduce dataset from Delta EE pose using IK.")
    parser.add_argument("input_file", type=str, help="Path to the source HDF5 file")
    parser.add_argument("--output_file", type=str, default="reproduced_delta_ee_episode.h5", help="Path to save")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found.")
        return

    print(f"🔄 Loading Model: {MODEL_PATH}")
    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Get IDs
    try:
        tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
        back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
        link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "6_Link")
        n_motors = model.nu
    except Exception as e:
        print(f"⚠️ Warning: Some IDs not found: {e}")
        link6_id = -1
        n_motors = 6

    # Load source data
    print(f"📖 Reading source data from {args.input_file}...")
    with h5py.File(args.input_file, 'r') as f:
        source_qpos = f['observations/qpos'][:] 
        source_action = f['action'][:] # Delta EE
        source_timestamp = f['timestamp'][:]
        source_phase = f['phase'][:]
        n_frames = len(source_qpos)

    recorder = ReplayRecorder(args.output_file)
    
    print("🚀 Starting Delta-EE-based Reproduction (IK with closed-loop accumulation)...")
    
    # 1. Initialize Frame 0 (Set exact start state)
    mujoco.mj_resetData(model, data)
    data.qpos[:n_motors] = np.deg2rad(source_qpos[0])
    mujoco.mj_forward(model, data)
    current_q_guess = np.deg2rad(source_qpos[0])
    
    # Measurements for Frame 0
    current_ee_pose_mm = get_ee_pose_6d_scaled(data, link6_id)
    p_sensor = data.site_xpos[tip_id].copy()
    needle_dir = (data.site_xpos[tip_id] - data.site_xpos[back_id])
    needle_dir /= (np.linalg.norm(needle_dir) + 1e-10)
    dist = mujoco.mj_ray(model, data, p_sensor, needle_dir, None, 1, link6_id, np.zeros(1, dtype=np.int32))
    current_sensor_dist = dist * 1000.0 if dist >= 0 else -1.0
    
    frames = {}
    for cam_name in ["side_camera", "tool_camera", "top_camera"]:
        renderer.update_scene(data, camera=cam_name)
        frames[cam_name] = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        
    recorder.add(frames, np.rad2deg(data.qpos[:n_motors]), current_ee_pose_mm, source_action[0], source_timestamp[0], source_phase[0], current_sensor_dist)

    # 2. Iterate
    for i in tqdm(range(1, n_frames), desc="Solving IK (Delta)"):
        # The action at index i is the delta from (i-1) to (i)
        delta = source_action[i]
        
        # Current actual pose in sim
        current_sim_ee = get_ee_pose_6d_scaled(data, link6_id)
        
        # Target = Current + Delta (Closed Loop)
        target_ee = current_sim_ee + delta
        
        target_pos_m = target_ee[:3] / 1000.0
        target_r, target_p, target_y = target_ee[3], target_ee[4], target_ee[5]
        target_rot_mat = euler_to_rot_mat(target_r, target_p, target_y)
        
        # Run IK
        q_sol, converged = solve_ik(model, data, target_pos_m, target_rot_mat, link6_id, current_q_guess, n_motors)
        current_q_guess = q_sol
        
        # Update
        data.qpos[:n_motors] = q_sol
        mujoco.mj_forward(model, data)
        
        # Measure
        current_ee_pose_mm = get_ee_pose_6d_scaled(data, link6_id)
        p_sensor = data.site_xpos[tip_id].copy()
        needle_dir = (data.site_xpos[tip_id] - data.site_xpos[back_id])
        needle_dir /= (np.linalg.norm(needle_dir) + 1e-10)
        dist = mujoco.mj_ray(model, data, p_sensor, needle_dir, None, 1, link6_id, np.zeros(1, dtype=np.int32))
        current_sensor_dist = dist * 1000.0 if dist >= 0 else -1.0
        
        frames = {}
        for cam_name in ["side_camera", "tool_camera", "top_camera"]:
            renderer.update_scene(data, camera=cam_name)
            frames[cam_name] = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            
        recorder.add(frames, np.rad2deg(data.qpos[:n_motors]), current_ee_pose_mm, source_action[i], source_timestamp[i], source_phase[i], current_sensor_dist)

    recorder.save()

if __name__ == "__main__":
    main()
