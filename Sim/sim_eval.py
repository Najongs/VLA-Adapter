"""
sim_eval.py

Evaluates a fine-tuned VLA-Adapter model in the MuJoCo needle-insertion simulation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python Sim/sim_eval.py \
        --pretrained_checkpoint outputs/YOUR_CHECKPOINT_DIR \
        --num_episodes 50

The script:
  1. Loads the fine-tuned VLA-Adapter checkpoint (VLM + LoRA + action_head + proprio_projector)
  2. Initializes the MuJoCo needle-insertion environment
  3. Runs closed-loop evaluation: render → model.predict_action → IK → mj_step
  4. Reports success rate and optionally saves rollout videos
"""

import os
os.environ['MUJOCO_GL'] = 'egl'

import sys
import argparse
import random
import pathlib

import mujoco
import numpy as np
import cv2
import torch
import imageio
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.robot.robot_utils import get_action, get_image_resize_size, set_seed_everywhere
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
    DEVICE,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════
SIM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "meca_add.xml")
TARGET_INSERTION_DEPTH = 0.0275
TASK_INSTRUCTION = "Align the needle and insert it into the trocar opening"
IMG_WIDTH, IMG_HEIGHT = 640, 480


def parse_args():
    parser = argparse.ArgumentParser(description="VLA-Adapter Sim Eval")
    # Model
    parser.add_argument("--pretrained_checkpoint", type=str, required=True)
    parser.add_argument("--use_proprio", type=str, default="True")
    parser.add_argument("--num_images_in_input", type=int, default=3)
    parser.add_argument("--use_film", type=str, default="False")
    parser.add_argument("--use_minivlm", type=str, default="True")
    parser.add_argument("--use_pro_version", type=str, default="True")
    parser.add_argument("--use_l1_regression", type=str, default="True")
    parser.add_argument("--center_crop", type=str, default="True")
    parser.add_argument("--model_family", type=str, default="openvla")
    parser.add_argument("--load_in_8bit", type=str, default="False")
    parser.add_argument("--load_in_4bit", type=str, default="False")
    parser.add_argument("--unnorm_key", type=str, default="needle_insertion")
    parser.add_argument("--base_vlm_path", type=str, default="pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b",
                        help="Path to base VLM (used when checkpoint only has LoRA adapter)")
    # Eval
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--max_steps_per_episode", type=int, default=600)
    parser.add_argument("--num_open_loop_steps", type=int, default=8)
    parser.add_argument("--sim_steps_per_control", type=int, default=67)
    parser.add_argument("--randomize_phantom_pos", type=str, default="True")
    parser.add_argument("--save_video", type=str, default="True")
    parser.add_argument("--video_fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # Convert string bools
    for k in ["use_proprio", "use_film", "use_minivlm", "use_pro_version",
              "use_l1_regression", "center_crop", "load_in_8bit", "load_in_4bit",
              "randomize_phantom_pos", "save_video"]:
        setattr(args, k, getattr(args, k) in ("True", "true", "1"))
    return args


# ═══════════════════════════════════════════════════════════════════════════════
# MuJoCo Environment (adapted from VLANeXt sim_eval.py / Save_dataset.py)
# ═══════════════════════════════════════════════════════════════════════════════
def smooth_step(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def randomize_phantom_pos(model_mj, data_mj, phantom_id, rot_id):
    offset_x = np.random.uniform(-0.05, 0.05)
    offset_y = np.random.uniform(-0.03, 0.03)
    model_mj.body_pos[phantom_id] = np.array([offset_x, offset_y, 0.0])
    random_angle_deg = np.random.uniform(-15, 15)
    new_quat = np.zeros(4)
    mujoco.mju_euler2Quat(new_quat, [0, 0, np.deg2rad(random_angle_deg)], "xyz")
    model_mj.body_quat[rot_id] = new_quat
    mujoco.mj_forward(model_mj, data_mj)
    return np.array([offset_x, offset_y, 0.0], dtype=np.float32), new_quat.astype(np.float32)


def project_to_2d(point_3d, model_mj, data_mj, cam_name, img_w, img_h):
    cam_id = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_pos = data_mj.cam_xpos[cam_id]
    cam_mat = data_mj.cam_xmat[cam_id].reshape(3, 3)
    fovy = model_mj.cam_fovy[cam_id]
    p_cam = cam_mat.T @ (point_3d - cam_pos)
    f = img_h / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
    u = -f * (p_cam[0] / p_cam[2]) + (img_w - 1) / 2.0
    v = f * (p_cam[1] / p_cam[2]) + (img_h - 1) / 2.0
    return np.array([u / img_w, v / img_h], dtype=np.float32)


class SimEnv:
    """MuJoCo needle-insertion environment for closed-loop evaluation."""

    def __init__(self, model_xml_path, randomize=True):
        self.model = mujoco.MjModel.from_xml_path(model_xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=IMG_HEIGHT, width=IMG_WIDTH)

        self.tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
        self.back_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
        self.target_entry_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "trocar_target")
        self.target_depth_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "trocar_depth")
        self.phantom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "phantom_assembly")
        self.rotating_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rotating_assembly")
        self.link6_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "6_Link")
        self.n_motors = self.model.nu
        self.dof = self.model.nv
        self.randomize = randomize

    def get_ee_pose(self):
        """Return (6,) array: [x, y, z] in mm, [rx, ry, rz] in rad."""
        pos = self.data.xpos[self.link6_id].copy() * 1000.0
        mat = self.data.xmat[self.link6_id].reshape(3, 3)
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

    def render_cameras(self):
        frames = {}
        for cam_name in ["side_camera", "tool_camera", "top_camera"]:
            self.renderer.update_scene(self.data, camera=cam_name)
            frames[cam_name] = self.renderer.render().copy()
        return frames

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        home_pose = np.array([
            np.random.uniform(-0.45, 0.55),
            np.random.uniform(-0.3, -0.4),
            np.random.uniform(0.3, 0.4),
            0.0,
            np.random.uniform(0.45, 0.55),
            np.random.uniform(0.95, 1.05),
        ])
        self.data.qpos[:6] = home_pose
        if self.randomize:
            randomize_phantom_pos(self.model, self.data, self.phantom_id, self.rotating_id)
        mujoco.mj_forward(self.model, self.data)

    def apply_delta_ee(self, delta_ee_6d, n_sim_steps=67, gain=0.5):
        """Apply predicted delta-EE (mm/rad) via resolved-rate IK."""
        current_ee = self.get_ee_pose()
        target_ee = current_ee + delta_ee_6d
        target_pos_m = target_ee[:3] / 1000.0
        target_rpy = target_ee[3:]

        for _ in range(n_sim_steps):
            cur_pos = self.data.xpos[self.link6_id].copy()
            cur_mat = self.data.xmat[self.link6_id].reshape(3, 3)
            err_pos = target_pos_m - cur_pos
            target_mat = self._rpy_to_rotmat(target_rpy)
            err_rot_mat = target_mat @ cur_mat.T
            err_rot = self._rotmat_to_axisangle(err_rot_mat)
            err = np.concatenate([err_pos * 50.0, err_rot * 10.0])
            jac_pos = np.zeros((3, self.dof))
            jac_rot = np.zeros((3, self.dof))
            mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, self.link6_id)
            J = np.vstack([jac_pos[:, :self.n_motors], jac_rot[:, :self.n_motors]])
            dq = np.linalg.pinv(J, rcond=1e-4) @ err
            self.data.ctrl[:self.n_motors] = self.data.qpos[:self.n_motors] + dq * gain
            mujoco.mj_step(self.model, self.data)

    def check_success(self, lateral_thresh=0.003, angle_thresh_deg=30.0):
        tip_pos = self.data.site_xpos[self.tip_id].copy()
        back_pos = self.data.site_xpos[self.back_id].copy()
        entry_pos = self.data.site_xpos[self.target_entry_id].copy()
        depth_pos = self.data.site_xpos[self.target_depth_id].copy()
        axis = depth_pos - entry_pos
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-8:
            return False
        axis_dir = axis / axis_len
        tip_offset = tip_pos - entry_pos
        depth = np.dot(tip_offset, axis_dir)
        if depth < TARGET_INSERTION_DEPTH:
            return False
        projection = tip_offset - depth * axis_dir
        if np.linalg.norm(projection) > lateral_thresh:
            return False
        needle_dir = tip_pos - back_pos
        needle_len = np.linalg.norm(needle_dir)
        if needle_len < 1e-8:
            return False
        needle_dir /= needle_len
        cos_angle = abs(np.dot(needle_dir, axis_dir))
        if np.degrees(np.arccos(np.clip(cos_angle, 0.0, 1.0))) > angle_thresh_deg:
            return False
        return True

    def get_sensor_dist(self):
        tip_pos = self.data.site_xpos[self.tip_id].copy()
        back_pos = self.data.site_xpos[self.back_id].copy()
        needle_dir = tip_pos - back_pos
        nd_len = np.linalg.norm(needle_dir)
        if nd_len > 1e-8:
            needle_dir /= nd_len
        dist = mujoco.mj_ray(
            self.model, self.data, tip_pos, needle_dir,
            None, 1, self.link6_id, np.zeros(1, dtype=np.int32),
        )
        return dist * 1000.0 if dist >= 0 else -1.0

    def get_spatial_metrics(self):
        tip_pos = self.data.site_xpos[self.tip_id].copy()
        back_pos = self.data.site_xpos[self.back_id].copy()
        entry_pos = self.data.site_xpos[self.target_entry_id].copy()
        depth_pos = self.data.site_xpos[self.target_depth_id].copy()
        dist_mm = np.linalg.norm((entry_pos - tip_pos) * 1000.0)
        axis = depth_pos - entry_pos
        axis_dir = axis / (np.linalg.norm(axis) + 1e-10)
        tip_offset = tip_pos - entry_pos
        insertion_depth_mm = np.dot(tip_offset, axis_dir) * 1000.0
        projection = tip_offset - np.dot(tip_offset, axis_dir) * axis_dir
        lateral_mm = np.linalg.norm(projection) * 1000.0
        needle_dir = tip_pos - back_pos
        needle_len = np.linalg.norm(needle_dir)
        if needle_len > 1e-8:
            needle_dir /= needle_len
            cos_angle = abs(np.dot(needle_dir, axis_dir))
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, 0.0, 1.0)))
        else:
            angle_deg = 90.0
        tip_uv = project_to_2d(tip_pos, self.model, self.data, "tool_camera", IMG_WIDTH, IMG_HEIGHT)
        trocar_uv = project_to_2d(entry_pos, self.model, self.data, "tool_camera", IMG_WIDTH, IMG_HEIGHT)
        return {
            "dist_mm": dist_mm, "insertion_depth_mm": insertion_depth_mm,
            "lateral_mm": lateral_mm, "angle_deg": angle_deg,
            "tip_uv": tip_uv, "trocar_uv": trocar_uv,
        }

    @staticmethod
    def _rpy_to_rotmat(rpy):
        r, p, y = rpy
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return Rz @ Ry @ Rx

    @staticmethod
    def _rotmat_to_axisangle(R):
        angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
        if abs(angle) < 1e-6:
            return np.zeros(3)
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(angle))
        return axis * angle


# ═══════════════════════════════════════════════════════════════════════════════
# Image preprocessing (match RLDS training pipeline)
# ═══════════════════════════════════════════════════════════════════════════════
def preprocess_image(img_rgb, target_size=224):
    """JPEG encode→decode + resize to match training distribution."""
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, 95])
    img_decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    img_rgb_out = cv2.cvtColor(img_decoded, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb_out, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    return img_resized


# ═══════════════════════════════════════════════════════════════════════════════
# Video / Plot saving
# ═══════════════════════════════════════════════════════════════════════════════
def save_rollout_video(images, episode_idx, success, save_dir, fps=15):
    os.makedirs(save_dir, exist_ok=True)
    tag = "success" if success else "fail"
    path = os.path.join(save_dir, f"episode_{episode_idx:04d}_{tag}.mp4")
    writer = imageio.get_writer(path, fps=fps)
    for img in images:
        writer.append_data(img)
    writer.close()
    print(f"  Saved video: {path}")


def draw_overlay(frame, metrics, ctrl_step):
    lines = [
        f"step={ctrl_step}",
        f"dist={metrics['dist_mm']:.1f}mm",
        f"depth={metrics['insertion_depth_mm']:.1f}mm",
        f"lateral={metrics['lateral_mm']:.1f}mm",
        f"angle={metrics['angle_deg']:.1f}deg",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (5, 14 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (5, 14 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def save_episode_plot(metrics_history, episode_idx, success, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    tag = "SUCCESS" if success else "FAIL"
    steps = list(range(len(metrics_history)))
    dist = [m["dist_mm"] for m in metrics_history]
    depth = [m["insertion_depth_mm"] for m in metrics_history]
    lateral = [m["lateral_mm"] for m in metrics_history]
    angle = [m["angle_deg"] for m in metrics_history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Episode {episode_idx} [{tag}]", fontsize=14, fontweight="bold")

    axes[0, 0].plot(steps, dist, "b-", linewidth=1.2)
    axes[0, 0].set_ylabel("Distance (mm)")
    axes[0, 0].set_title("Needle Tip <-> Trocar Entry")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, depth, "g-", linewidth=1.2)
    axes[0, 1].axhline(y=TARGET_INSERTION_DEPTH * 1000, color="r", linestyle="--",
                        label=f"target={TARGET_INSERTION_DEPTH*1000:.1f}mm")
    axes[0, 1].set_ylabel("Depth (mm)")
    axes[0, 1].set_title("Insertion Depth (along axis)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(steps, lateral, "m-", linewidth=1.2)
    axes[1, 0].axhline(y=3.0, color="r", linestyle="--", label="thresh=3mm")
    axes[1, 0].set_ylabel("Lateral (mm)")
    axes[1, 0].set_title("Lateral Distance from Axis")
    axes[1, 0].set_xlabel("Control Step")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(steps, angle, "c-", linewidth=1.2)
    axes[1, 1].axhline(y=30.0, color="r", linestyle="--", label="thresh=30deg")
    axes[1, 1].set_ylabel("Angle (deg)")
    axes[1, 1].set_title("Needle-Trocar Axis Angle")
    axes[1, 1].set_xlabel("Control Step")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"episode_{episode_idx:04d}_{tag.lower()}_metrics.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading (LoRA checkpoint)
# ═══════════════════════════════════════════════════════════════════════════════
def load_vla_adapter(cfg):
    """Load VLA-Adapter: base VLM + LoRA adapter + dataset stats.

    For LoRA-only checkpoints (no config.json / model.safetensors),
    we load the base VLM first, then apply the LoRA adapter on top.
    If the checkpoint has a config.json (merged checkpoint), we fall back
    to the standard get_model() path.
    """
    import json
    from peft import PeftModel
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from experiments.robot.openvla_utils import update_auto_map, check_model_logic_mismatch

    ckpt_dir = cfg.pretrained_checkpoint
    adapter_dir = os.path.join(ckpt_dir, "lora_adapter")
    has_config = os.path.isfile(os.path.join(ckpt_dir, "config.json"))
    has_lora = os.path.isdir(adapter_dir) and os.path.isfile(os.path.join(adapter_dir, "adapter_config.json"))

    if has_config and not has_lora:
        # Merged checkpoint — use standard loading
        from experiments.robot.robot_utils import get_model
        return get_model(cfg)

    if not has_lora:
        raise FileNotFoundError(
            f"Checkpoint at {ckpt_dir} has neither config.json (merged) "
            f"nor lora_adapter/ (LoRA). Cannot load."
        )

    # ── LoRA-only checkpoint: load base VLM then apply adapter ────────────
    print(f"LoRA checkpoint detected. Loading base VLM from: {cfg.base_vlm_path}")

    # Register custom classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # For minivlm: load Prismatic VLM checkpoint, create empty HF model, transfer weights
    if cfg.use_minivlm:
        from prismatic.models import load as prismatic_load

        print(f"Loading Prismatic VLM from: {cfg.base_vlm_path}")
        vlm = prismatic_load(cfg.base_vlm_path, hf_token="", load_for_training=False)

        config_path = os.path.join("pretrained_models", "configs", "config.json")
        config = AutoConfig.from_pretrained(config_path)
        base_vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)

        # Rename state dict keys: Prismatic → HF model naming
        replace_map = [
            ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
            ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
            ("llm_backbone.llm", "language_model"),
            ("projector.projector.0", "projector.fc1"),
            ("projector.projector.2", "projector.fc2"),
            ("projector.projector.4", "projector.fc3"),
            ("gamma", "scale_factor"),
        ]
        old_sd = vlm.state_dict()
        new_sd = {}
        for k, v in old_sd.items():
            new_k = k
            for old, new in replace_map:
                if old in new_k:
                    new_k = new_k.replace(old, new)
            new_sd[new_k] = v

        missing, unexpected = base_vla.load_state_dict(new_sd, strict=False)
        print(f"Base VLM loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        del vlm, old_sd, new_sd
    else:
        config_path = "pretrained_models/configs"
        update_auto_map(config_path)
        check_model_logic_mismatch(config_path)
        base_vla = AutoModelForVision2Seq.from_pretrained(
            config_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
        )

    # Apply LoRA adapter
    print(f"Applying LoRA adapter from: {adapter_dir}")
    vla = PeftModel.from_pretrained(base_vla, adapter_dir)
    vla = vla.merge_and_unload()

    # Set num images
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    vla.eval()
    vla = vla.to(DEVICE)

    # Load dataset statistics for action denormalization
    stats_path = os.path.join(ckpt_dir, "dataset_statistics.json")
    if os.path.isfile(stats_path):
        with open(stats_path, "r") as f:
            vla.norm_stats = json.load(f)
        print(f"Loaded dataset statistics from: {stats_path}")
    else:
        print(f"WARNING: No dataset_statistics.json found at {ckpt_dir}")

    return vla


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    cfg = parse_args()

    # Seed
    set_seed_everywhere(cfg.seed)

    # ── Load VLA-Adapter model + components ──────────────────────────────────
    print(f"Loading VLA-Adapter from: {cfg.pretrained_checkpoint}")
    model = load_vla_adapter(cfg)
    model.set_version("vla-adapter")

    # Load processor from checkpoint dir (has tokenizer etc.)
    processor = get_processor(cfg)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    action_head = None
    if cfg.use_l1_regression:
        action_head = get_action_head(cfg, model.llm_dim)

    # Check unnorm_key
    if cfg.unnorm_key not in model.norm_stats:
        available = list(model.norm_stats.keys())
        print(f"WARNING: unnorm_key '{cfg.unnorm_key}' not found. Available: {available}")
        if len(available) == 1:
            cfg.unnorm_key = available[0]
            print(f"Auto-selected: {cfg.unnorm_key}")
        else:
            # Try to find a needle-related key
            for k in available:
                if "needle" in k:
                    cfg.unnorm_key = k
                    print(f"Auto-selected: {cfg.unnorm_key}")
                    break

    resize_size = get_image_resize_size(cfg)  # 224 for openvla
    print(f"Image resize: {resize_size}, unnorm_key: {cfg.unnorm_key}")
    print(f"Action chunk size: {NUM_ACTIONS_CHUNK}, open-loop steps: {cfg.num_open_loop_steps}")

    # ── Output directory ─────────────────────────────────────────────────────
    ckpt_path = pathlib.Path(cfg.pretrained_checkpoint)
    eval_dir = ckpt_path / f"sim_eval_ep{cfg.num_episodes}_exec{cfg.num_open_loop_steps}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_path = eval_dir / "log.txt"
    log_file = open(log_path, "w")
    print(f"Logging to {log_path}")

    # ── Environment ──────────────────────────────────────────────────────────
    model_xml = os.path.abspath(SIM_MODEL_PATH)
    env = SimEnv(model_xml, randomize=cfg.randomize_phantom_pos)

    total_successes = 0

    # CSV summary
    csv_path = eval_dir / "metrics_summary.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "success", "steps", "final_dist_mm", "final_depth_mm",
                         "final_lateral_mm", "final_angle_deg", "min_dist_mm"])

    from collections import deque

    for ep in range(1, cfg.num_episodes + 1):
        env.reset()
        last_ee_pose = env.get_ee_pose()

        action_queue = deque(maxlen=cfg.num_open_loop_steps)
        replay_images = []
        metrics_history = []
        success = False

        for ctrl_step in range(cfg.max_steps_per_episode):
            # ── 1. Observe ───────────────────────────────────────────────────
            frames = env.render_cameras()
            # side_camera → primary (full_image)
            # tool_camera → wrist
            # top_camera  → secondary (extra wrist in VLA-Adapter's convention)
            img_primary = preprocess_image(frames["side_camera"], resize_size)
            img_wrist = preprocess_image(frames["tool_camera"], resize_size)
            img_top = preprocess_image(frames["top_camera"], resize_size)

            # Spatial metrics
            metrics = env.get_spatial_metrics()
            metrics_history.append(metrics)

            # Video replay frame (3 views side by side, use 256px for visibility)
            replay_ext = cv2.resize(frames["side_camera"], (256, 256))
            replay_wrist = cv2.resize(frames["tool_camera"], (256, 256))
            replay_top = cv2.resize(frames["top_camera"], (256, 256))
            replay_frame = np.concatenate([replay_ext, replay_wrist, replay_top], axis=1)
            draw_overlay(replay_frame, metrics, ctrl_step)
            replay_images.append(replay_frame)

            # Proprioception: ee_pose(6) + gripper_proxy(1) + gripper_proxy(1) = 8D
            # (matches training data: ee_pose[:6] + gripper_state duplicated to 8D)
            ee_pose = env.get_ee_pose()  # (6,)
            sensor_dist = env.get_sensor_dist()
            gripper_state = 1.0 if (0 < sensor_dist < 30) else 0.0
            # proprio is 8D: ee_pose(6) + gripper(1) + gripper(1) to match RLDS conversion
            proprio = np.concatenate([ee_pose, [gripper_state, gripper_state]]).astype(np.float32)

            # Build observation dict matching VLA-Adapter's expected format
            observation = {
                "full_image": img_primary,
                "wrist_image": img_wrist,
                "wrist_image_2": img_top,  # secondary camera as extra "wrist"
                "state": proprio,
            }

            # ── 2. Get action from model (or from buffer) ────────────────────
            if len(action_queue) == 0:
                actions = get_action(
                    cfg, model, observation, TASK_INSTRUCTION,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    use_film=cfg.use_film,
                    use_minivlm=cfg.use_minivlm,
                )
                # actions is a list of np arrays, each (7,) = delta_ee(6) + gripper(1)
                # Already denormalized by predict_action
                action_queue.extend(actions)

            raw_action = action_queue.popleft()  # (7,) denormalized
            delta_ee = raw_action[:6]  # (dx, dy, dz, drx, dry, drz) in mm/rad

            # ── 3. Apply action ──────────────────────────────────────────────
            env.apply_delta_ee(delta_ee, n_sim_steps=cfg.sim_steps_per_control)
            last_ee_pose = env.get_ee_pose()

            # ── 4. Check success ─────────────────────────────────────────────
            if env.check_success():
                success = True
                metrics_history.append(env.get_spatial_metrics())
                break

        if success:
            total_successes += 1

        sr = total_successes / ep * 100
        final_m = metrics_history[-1]
        min_dist = min(m["dist_mm"] for m in metrics_history)
        msg = (f"Episode {ep}/{cfg.num_episodes} | {'SUCCESS' if success else 'FAIL'} | "
               f"Steps: {ctrl_step + 1} | SR: {sr:.1f}% ({total_successes}/{ep}) | "
               f"dist={final_m['dist_mm']:.1f}mm lateral={final_m['lateral_mm']:.1f}mm "
               f"angle={final_m['angle_deg']:.1f}deg min_dist={min_dist:.1f}mm")
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

        csv_writer.writerow([
            ep, int(success), ctrl_step + 1,
            f"{final_m['dist_mm']:.2f}", f"{final_m['insertion_depth_mm']:.2f}",
            f"{final_m['lateral_mm']:.2f}", f"{final_m['angle_deg']:.2f}",
            f"{min_dist:.2f}",
        ])
        csv_file.flush()

        save_episode_plot(metrics_history, ep, success, str(eval_dir))

        if cfg.save_video:
            save_rollout_video(replay_images, ep, success, str(eval_dir), fps=cfg.video_fps)

    csv_file.close()

    final_sr = total_successes / cfg.num_episodes * 100
    summary = f"\n{'='*60}\nFinal Success Rate: {final_sr:.2f}% ({total_successes}/{cfg.num_episodes})\n{'='*60}"
    print(summary)
    log_file.write(summary + "\n")
    log_file.close()

    # Rename directory to include success rate
    new_dir = eval_dir.parent / f"{eval_dir.name}_SR{final_sr:.2f}"
    try:
        eval_dir.rename(new_dir)
        print(f"Results saved to: {new_dir}")
    except OSError:
        print(f"Results saved to: {eval_dir}")


if __name__ == "__main__":
    main()
