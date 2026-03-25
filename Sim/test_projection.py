"""
Test script to verify MuJoCo 3D→2D camera projection + spatial target values.

Runs a full expert episode (align → insert) and saves annotated wrist camera
images every ~100 save-steps, printing all spatial target values.

Usage:
    cd Sim && python test_projection.py
"""

import os
os.environ['MUJOCO_GL'] = 'egl'

import mujoco
import numpy as np
import cv2

MODEL_PATH = "meca_add.xml"
IMG_WIDTH = 640
IMG_HEIGHT = 480
OUTPUT_DIR = "projection_test_output"
TARGET_INSERTION_DEPTH = 0.0275
TRAJ_DURATION = 15.0


def project_to_2d(point_3d, model, data, cam_name, img_w, img_h):
    """Project a 3D world point to normalized 2D image coordinates [0,1]."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    fovy = model.cam_fovy[cam_id]

    p_cam = cam_mat.T @ (point_3d - cam_pos)

    f = img_h / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
    u = -f * (p_cam[0] / p_cam[2]) + (img_w - 1) / 2.0
    v =  f * (p_cam[1] / p_cam[2]) + (img_h - 1) / 2.0

    return np.array([u / img_w, v / img_h], dtype=np.float32)


def smooth_step(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading model: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)

    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
    back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
    target_entry_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_target")
    target_depth_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_depth")
    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "6_Link")
    n_motors = model.nu
    dof = model.nv

    # --- Reset ---
    mujoco.mj_resetData(model, data)
    home_pose = np.array([np.random.uniform(-0.45, 0.55) , np.random.uniform(-0.3, -0.4), np.random.uniform(0.3, 0.4),  0.0000,np.random.uniform(0.45, 0.55), np.random.uniform(0.95, 1.05)]) # (30, -20, 20, 0, 30, 60)
    data.qpos[:6] = home_pose
    mujoco.mj_forward(model, data)

    p_entry = data.site_xpos[target_entry_id].copy()
    p_depth = data.site_xpos[target_depth_id].copy()
    start_tip = data.site_xpos[tip_id].copy()
    start_back = data.site_xpos[back_id].copy()
    needle_len = np.linalg.norm(start_tip - start_back)

    task_state = 1
    traj_start_time = data.time
    traj_initialized = False
    insertion_started = False
    accumulated_depth = 0.0
    align_timer = 0
    hold_start_time = None
    current_speed = 0.45

    step_count = 0
    save_count = 0

    print(f"\n{'='*90}")
    print(f"{'Save':>5} | {'Phase':>5} | {'tip_u':>6} {'tip_v':>6} | {'tro_u':>6} {'tro_v':>6} | "
          f"{'tip_vis':>7} {'tro_vis':>7} | {'dist_mm':>8} | {'phase_bin':>9}")
    print(f"{'='*90}")

    while True:
        curr_tip = data.site_xpos[tip_id].copy()
        curr_back = data.site_xpos[back_id].copy()

        # --- Expert trajectory (same as Save_dataset.py) ---
        if task_state == 1:
            if not traj_initialized:
                traj_start_time = data.time
                start_tip_pos = curr_tip.copy()
                start_back_pos = curr_back.copy()
                traj_initialized = True
            progress = smooth_step((data.time - traj_start_time) / TRAJ_DURATION)
            axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
            goal_tip = p_entry - (axis_dir * 0.0001)
            goal_back = p_entry - (axis_dir * (0.0001 + needle_len))
            target_tip_pos = (1 - progress) * start_tip_pos + progress * goal_tip
            target_back_pos = (1 - progress) * start_back_pos + progress * goal_back
            if progress >= 1.0:
                if np.linalg.norm(curr_tip - goal_tip) < 0.002:
                    align_timer += 1
                else:
                    align_timer = 0
                if align_timer > 20:
                    task_state = 2
                    insertion_started = False

        elif task_state == 2:
            if not insertion_started:
                phase3_base_tip = curr_tip.copy()
                insertion_started = True
                accumulated_depth = 0.0
                hold_start_time = None
            axis_dir = (p_depth - p_entry) / (np.linalg.norm(p_depth - p_entry) + 1e-10)
            if accumulated_depth < TARGET_INSERTION_DEPTH:
                accumulated_depth += 0.0000025
                target_tip_pos = phase3_base_tip + (axis_dir * accumulated_depth)
                target_back_pos = target_tip_pos - (axis_dir * needle_len)
                if accumulated_depth >= TARGET_INSERTION_DEPTH:
                    hold_start_time = data.time
            else:
                if hold_start_time is None:
                    hold_start_time = data.time
                target_tip_pos = phase3_base_tip + (axis_dir * TARGET_INSERTION_DEPTH)
                target_back_pos = target_tip_pos - (axis_dir * needle_len)
                if data.time - hold_start_time >= 1.0:
                    break

        # --- IK ---
        err_tip = target_tip_pos - curr_tip
        err_back = target_back_pos - curr_back
        tip_rot_mat = data.site_xmat[tip_id].reshape(3, 3)
        offset_angle = np.deg2rad(210)
        offset_local_vec = np.array([np.cos(offset_angle), np.sin(offset_angle), 0])
        current_side_vec = tip_rot_mat @ offset_local_vec
        needle_axis_curr = (curr_tip - curr_back) / (np.linalg.norm(curr_tip - curr_back) + 1e-10)
        target_side_vec = np.cross(needle_axis_curr, np.array([0, 0, 1]))
        target_side_vec = target_side_vec / np.linalg.norm(target_side_vec) if np.linalg.norm(target_side_vec) > 1e-3 else np.array([1, 0, 0])
        err_roll = np.cross(current_side_vec, target_side_vec)

        jac_tip_full = np.zeros((6, dof))
        jac_back = np.zeros((3, dof))
        mujoco.mj_jacSite(model, data, jac_tip_full[:3], jac_tip_full[3:], tip_id)
        mujoco.mj_jacSite(model, data, jac_back, None, back_id)

        J_p1 = jac_tip_full[:3, :n_motors]
        e_p1 = err_tip * 50.0
        if np.linalg.norm(e_p1) > 1.0:
            e_p1 = e_p1 / np.linalg.norm(e_p1) * 1.0
        J_p1_pinv = np.linalg.pinv(J_p1, rcond=1e-4)
        dq_p1 = J_p1_pinv @ e_p1
        P_null_1 = np.eye(n_motors) - (J_p1_pinv @ J_p1)
        J_p2_proj = jac_back[:, :n_motors] @ P_null_1
        dq_p2 = np.linalg.pinv(J_p2_proj, rcond=1e-4) @ ((err_back * 50.0) - jac_back[:, :n_motors] @ dq_p1)
        P_null_2 = P_null_1 - (np.linalg.pinv(J_p2_proj, rcond=1e-4) @ J_p2_proj)
        J_p3_proj = jac_tip_full[3:, :n_motors] @ P_null_2
        dq_p3 = np.linalg.pinv(J_p3_proj, rcond=1e-4) @ ((err_roll * 10.0) - jac_tip_full[3:, :n_motors] @ (dq_p1 + dq_p2))

        data.ctrl[:n_motors] = data.qpos[:n_motors] + (dq_p1 + dq_p2 + dq_p3) * current_speed
        mujoco.mj_step(model, data)
        step_count += 1

        # --- Save every 67 sim steps (same interval as Save_dataset.py) ---
        if step_count % 67 == 0:
            # Render at save_count=0 and every 100 save-steps
            if save_count != 0 and save_count % 100 != 0:
                save_count += 1
                continue

            # Compute spatial target values
            needle_tip_mm = data.site_xpos[tip_id].copy() * 1000
            trocar_entry_mm = data.site_xpos[target_entry_id].copy() * 1000

            tip_uv = project_to_2d(data.site_xpos[tip_id], model, data, "tool_camera", IMG_WIDTH, IMG_HEIGHT)
            trocar_uv = project_to_2d(data.site_xpos[target_entry_id], model, data, "tool_camera", IMG_WIDTH, IMG_HEIGHT)

            tip_vis = 1.0 if (0.0 <= tip_uv[0] <= 1.0 and 0.0 <= tip_uv[1] <= 1.0) else 0.0
            trocar_vis = 1.0 if (0.0 <= trocar_uv[0] <= 1.0 and 0.0 <= trocar_uv[1] <= 1.0) else 0.0

            dist_mm = np.linalg.norm(trocar_entry_mm - needle_tip_mm)
            dist_norm = dist_mm / 100.0
            phase_binary = max(0.0, min(task_state - 1, 1.0))

            print(f"{save_count:5d} | {task_state:5d} | {tip_uv[0]:6.3f} {tip_uv[1]:6.3f} | "
                  f"{trocar_uv[0]:6.3f} {trocar_uv[1]:6.3f} | "
                  f"{tip_vis:7.0f} {trocar_vis:7.0f} | {dist_mm:8.2f} | {phase_binary:9.0f}")

            # Render wrist camera with annotations
            renderer.update_scene(data, camera="tool_camera")
            img = renderer.render().copy()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Draw keypoints (only if visible)
            if tip_vis:
                px = (int(tip_uv[0] * IMG_WIDTH), int(tip_uv[1] * IMG_HEIGHT))
                cv2.circle(img_bgr, px, 8, (0, 0, 255), 2)
                cv2.putText(img_bgr, "needle_tip", (px[0]+10, px[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if trocar_vis:
                px = (int(trocar_uv[0] * IMG_WIDTH), int(trocar_uv[1] * IMG_HEIGHT))
                cv2.circle(img_bgr, px, 8, (0, 255, 0), 2)
                cv2.putText(img_bgr, "trocar", (px[0]+10, px[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Info overlay
            info_lines = [
                f"save_step={save_count}  phase={task_state}",
                f"tip_uv=({tip_uv[0]:.3f}, {tip_uv[1]:.3f})  vis={tip_vis:.0f}",
                f"trocar_uv=({trocar_uv[0]:.3f}, {trocar_uv[1]:.3f})  vis={trocar_vis:.0f}",
                f"dist={dist_mm:.1f}mm  dist_norm={dist_norm:.3f}  phase_bin={phase_binary:.0f}",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(img_bgr, line, (10, 20 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            out_path = os.path.join(OUTPUT_DIR, f"save{save_count:04d}_wrist.png")
            cv2.imwrite(out_path, img_bgr)

            save_count += 1

        if data.time - traj_start_time > 50.0:
            print("Timeout!")
            break

    print(f"{'='*90}")
    print(f"Done. Total sim steps={step_count}, save steps={save_count}")
    print(f"Check {OUTPUT_DIR}/ for annotated wrist camera images.")


if __name__ == "__main__":
    main()
