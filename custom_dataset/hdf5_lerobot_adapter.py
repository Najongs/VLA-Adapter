"""
HDF5 LeRobot Adapter for New VLA Dataset

This module provides an adapter to convert HDF5-format VLA dataset to LeRobot format
for training SmolVLA and other vision-language-action models.
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from torch.utils.data import Dataset, ConcatDataset
import logging
import random
import cv2

logger = logging.getLogger(__name__)


def normalize_angle_delta(delta: np.ndarray) -> np.ndarray:
    """
    Normalize angle delta to [-π, π] to handle wrapping.

    Args:
        delta: Angle delta in radians

    Returns:
        Normalized angle delta in [-π, π]
    """
    return np.arctan2(np.sin(delta), np.cos(delta))


class HDF5LeRobotDataset(Dataset):
    """
    Adapter that loads HDF5 VLA episodes and provides LeRobot-compatible samples.

    HDF5 structure:
    - action: (N, 6) - delta pose actions
    - observations/
        - ee_pose: (N, 6) - end-effector poses
        - qpos: (N, 6) - joint positions
        - images/
            - camera1: (N, 480, 640, 3)
            - camera2: (N, 480, 640, 3)
            - camera3: (N, 480, 640, 3)
    - timestamp: (N,)
    - phase: (N,) - task phase (1=Align, 2=Insert, 3=Hold)

    LeRobot expects samples with:
    {
        "observation.images.camera1": Tensor (C, H, W),
        "observation.images.camera2": Tensor (C, H, W),
        "observation.images.camera3": Tensor (C, H, W),
        "observation.state": Tensor (state_dim,),
        "action": Tensor (action_dim,),
        "task": str,
        "phase": str,  # "align", "insert", or "hold"
        "phase_id": int,  # 1, 2, or 3
        "timestamp": float,
        "frame_index": int,
        "episode_index": int,
        "index": int,
    }
    """

    def __init__(
        self,
        hdf5_path: str,
        episode_index: int = 0,
        horizon: int = 1,
        n_obs_steps: int = 1,
        use_qpos: bool = False,
        use_ee_pose: bool = True,
        use_sensor: bool = False,
        use_ee_pose_delta_as_action: bool = False,
        task_instruction: str = "Insert the needle into the target point",
        task_instruction1: str = "Move the needle to the eye phantom",
        task_instruction2: str = "insert the needle through the eye phantom trocar opening",
        cameras: Optional[List[str]] = None,
        camera_dropout_prob: float = 0.0,
        min_cameras: int = 1,
        augment: bool = True,
        augment_brightness: float = 0.2,
        augment_contrast: float = 0.2,
        augment_saturation: float = 0.2,
        augment_hue: float = 0.05,
        augment_noise: float = 0.02,
        squeeze_n_obs_steps: bool = False,
        tokenizer: Optional[Any] = None,
        tokenizer_max_length: int = 48,
        normalizer: Optional[Any] = None,
    ):
        """
        Args:
            hdf5_path: Path to HDF5 episode file
            episode_index: Episode index for this dataset
            horizon: Action prediction horizon (for future multi-step actions)
            n_obs_steps: Number of observation steps (temporal stacking)
            use_qpos: Use joint positions for state (6 dims)
            use_ee_pose: Use end-effector pose for state (6 dims, default)
            use_ee_pose_delta_as_action: Calculate action from ee_pose delta
            task_instruction: Task instruction text
            cameras: List of camera names to use (e.g., ["camera1", "camera3"]). None = use all cameras
            camera_dropout_prob: Probability of dropping out cameras (0.0 = disabled)
            min_cameras: Minimum number of cameras to keep active
            augment: Enable image augmentation
            augment_brightness: Max brightness adjustment factor
            augment_contrast: Max contrast adjustment factor
            augment_saturation: Max saturation adjustment factor
            augment_hue: Max hue adjustment (in degrees / 360)
            augment_noise: Gaussian noise std deviation
            squeeze_n_obs_steps: Squeeze temporal dimension if n_obs_steps=1 (for ACT)
            tokenizer: HuggingFace tokenizer for language conditioning (optional)
            tokenizer_max_length: Maximum length for tokenized text
            normalizer: Normalizer instance for action/state normalization (optional)
        """
        super().__init__()

        # Store normalizer
        self.normalizer = normalizer

        self.hdf5_path = Path(hdf5_path)
        self.episode_index = episode_index
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.squeeze_n_obs_steps = squeeze_n_obs_steps
        self.use_qpos = use_qpos
        self.use_ee_pose = use_ee_pose
        self.use_sensor = use_sensor
        self.use_ee_pose_delta_as_action = use_ee_pose_delta_as_action

        # Tokenizer for language conditioning (will be used after loading HDF5)
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        # Store task instructions
        self.default_task_instruction = task_instruction
        self.task_instruction1 = task_instruction1
        self.task_instruction2 = task_instruction2

        # Augmentation settings
        self.camera_dropout_prob = camera_dropout_prob
        self.min_cameras = min_cameras
        self.augment = augment
        self.augment_brightness = augment_brightness
        self.augment_contrast = augment_contrast
        self.augment_saturation = augment_saturation
        self.augment_hue = augment_hue
        self.augment_noise = augment_noise

        # Lazy HDF5 loading: open temporarily to read metadata, then close
        # With 30k+ episodes, keeping all files open causes RAM OOM
        self.h5file = None  # Will be opened on-demand in __getitem__

        with h5py.File(str(self.hdf5_path), 'r') as h5f:
            # Get dataset shapes
            self.num_frames = h5f['action'].shape[0]

            # Get actual camera names from HDF5 file
            all_actual_camera_names = sorted(list(h5f['observations']['images'].keys()))

            # Detect image format
            self.is_jpeg_format = self._detect_image_format_from(h5f, all_actual_camera_names)

            # Check if phase information is available
            self.has_phase = 'phase' in h5f

        # Tokenize task instructions
        if self.tokenizer is not None:
            # Helper to tokenize a string
            def tokenize_str(text):
                out = self.tokenizer(
                    text,
                    max_length=self.tokenizer_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                return out["input_ids"].squeeze(0), out["attention_mask"].squeeze(0)

            # Tokenize default and specific instructions
            self.default_tokens, self.default_mask = tokenize_str(self.default_task_instruction)
            self.task1_tokens, self.task1_mask = tokenize_str(self.task_instruction1)
            self.task2_tokens, self.task2_mask = tokenize_str(self.task_instruction2)
        else:
            self.default_tokens = self.default_mask = None
            self.task1_tokens = self.task1_mask = None
            self.task2_tokens = self.task2_mask = None

        # Create full mapping from camera index to actual camera name
        # Expected: camera1, camera2, camera3
        # Actual (sim): side_camera, tool_camera, top_camera
        full_camera_mapping = {}
        for cam_idx, actual_name in enumerate(all_actual_camera_names, start=1):
            expected_name = f"camera{cam_idx}"
            full_camera_mapping[expected_name] = actual_name

        # Filter cameras based on selection
        if cameras is None:
            # Use all cameras
            self.selected_cameras = [f"camera{i}" for i in range(1, len(all_actual_camera_names) + 1)]
        else:
            # Validate and use selected cameras
            self.selected_cameras = []
            for cam in cameras:
                if cam in full_camera_mapping:
                    self.selected_cameras.append(cam)
                else:
                    logger.warning(f"Camera '{cam}' not found in HDF5 file. Available: {list(full_camera_mapping.keys())}")

            if len(self.selected_cameras) == 0:
                raise ValueError(f"No valid cameras selected! Available cameras: {list(full_camera_mapping.keys())}")

        # Create mapping for selected cameras only
        self.camera_name_mapping = {cam: full_camera_mapping[cam] for cam in self.selected_cameras}
        self.num_cameras = len(self.selected_cameras)

        # Determine state dimension
        self.state_dim = 0
        if use_qpos:
            self.state_dim += 6
        if use_ee_pose:
            self.state_dim += 6
        # Sensor is added AFTER normalization, but counts towards total dimension
        if use_sensor:
            self.state_dim += 1

        # Phase mapping: numeric ID -> string label
        self.phase_mapping = {
            1: "align",
            2: "insert",
            3: "hold"
        }

        # Reduce logging verbosity - only log summary
        if episode_index == 0:
            logger.info(f"Loading HDF5 episodes from {self.hdf5_path.parent}")
            logger.info(f"  Selected cameras: {self.selected_cameras} (Total: {self.num_cameras})")
            logger.info(f"  State: {self.state_dim}D | Format: {'JPEG' if self.is_jpeg_format else 'GZIP'}")
            logger.info(f"  Camera mapping: {self.camera_name_mapping}")
            logger.info(f"  Phase info available: {self.has_phase}")
            if self.use_ee_pose_delta_as_action:
                logger.info("  Action mode: 'use_ee_pose_delta_as_action' is ENABLED")


    def _generate_instruction_from_phase(self) -> str:
        """
        Generate task instruction from phase information in HDF5.

        The dataset contains 3 phases:
        - Phase 1: Align needle with target point
        - Phase 2: Insert needle into target point
        - Phase 3: Hold needle at insertion depth

        This method analyzes the phase distribution and generates an appropriate instruction.

        Returns:
            Generated instruction string
        """
        if 'phase' not in self.h5file:
            return self.default_task_instruction

        # Read all phases
        phases = self.h5file['phase'][:]

        # Count phase occurrences
        unique_phases, counts = np.unique(phases, return_counts=True)

        # Determine dominant phase (most common)
        dominant_phase = unique_phases[np.argmax(counts)]

        # Phase-to-instruction mapping (3-state needle insertion task)
        phase_instructions = {
            1: "Align the needle with the target insertion point",
            2: "Insert the needle into the target point",
            3: "Hold the needle at the target insertion depth"
        }

        # If episode has multiple phases, use a general instruction
        if len(unique_phases) > 1:
            # Multi-phase episode: use general task description
            return "Insert the needle into the target point and hold"
        else:
            # Single-phase episode: use specific phase instruction
            return phase_instructions.get(dominant_phase, self.default_task_instruction)

    def _detect_image_format(self) -> bool:
        """Legacy method - opens file to detect format."""
        with h5py.File(str(self.hdf5_path), 'r') as h5f:
            cam_names = sorted(list(h5f['observations']['images'].keys()))
            return self._detect_image_format_from(h5f, cam_names)

    @staticmethod
    def _detect_image_format_from(h5f, camera_names: list) -> bool:
        """
        Detect whether images are stored in JPEG (compressed) or GZIP (uncompressed) format.

        Returns:
            True if JPEG format, False if GZIP format
        """
        try:
            # Get first camera dataset
            images_grp = h5f['observations']['images']
            first_cam = camera_names[0]
            first_dataset = images_grp[first_cam]

            # Read first frame to detect format
            sample = first_dataset[0]

            if isinstance(sample, np.ndarray):
                # Check shape to determine format
                if len(sample.shape) == 3 and sample.shape[2] == 3:
                    # Shape is (H, W, 3) → GZIP (already decoded image)
                    return False
                elif len(sample.shape) == 1 and sample.dtype == np.uint8:
                    # Shape is (N,) with uint8 → JPEG (compressed binary)
                    return True
                else:
                    # Fallback: check dataset shape
                    return len(first_dataset.shape) != 4
            else:
                # Not a numpy array → likely JPEG
                return True

        except Exception as e:
            logger.warning(f"Format detection failed: {e}, assuming GZIP format")
            return False

    def __len__(self) -> int:
        # Subtract (n_obs_steps - 1) for temporal stacking and (horizon - 1) for action sequences
        # We need at least n_obs_steps frames to create temporal observations
        effective_num_frames = self.num_frames
        if self.use_ee_pose_delta_as_action:
            # Need t+1 to compute action for frame t, so we lose one sample.
            effective_num_frames -= 1
        return max(0, effective_num_frames - max(self.n_obs_steps - 1, 0) - self.horizon + 1)

    def apply_image_augmentation(self, img_np: np.ndarray) -> np.ndarray:
        """
        Apply image augmentation to a single image.

        Args:
            img_np: Image array in [0, 1] range, shape (H, W, C)

        Returns:
            Augmented image in [0, 1] range
        """
        if not self.augment:
            return img_np

        # Convert to uint8 for cv2 operations
        img_uint8 = (img_np * 255).astype(np.uint8)

        # Random brightness and contrast adjustment
        if self.augment_brightness > 0 or self.augment_contrast > 0:
            alpha = 1.0 + random.uniform(-self.augment_contrast, self.augment_contrast)  # Contrast
            beta = random.uniform(-self.augment_brightness, self.augment_brightness) * 255  # Brightness
            img_uint8 = cv2.convertScaleAbs(img_uint8, alpha=alpha, beta=beta)

        # Random color augmentation (HSV)
        if self.augment_saturation > 0 or self.augment_hue > 0:
            img_hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)

            # Adjust hue
            if self.augment_hue > 0:
                hue_shift = random.uniform(-self.augment_hue, self.augment_hue) * 180
                img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_shift) % 180

            # Adjust saturation
            if self.augment_saturation > 0:
                sat_scale = 1.0 + random.uniform(-self.augment_saturation, self.augment_saturation)
                img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * sat_scale, 0, 255)

            img_uint8 = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Convert back to float [0, 1]
        img_float = img_uint8.astype(np.float32) / 255.0

        # Add Gaussian noise
        if self.augment_noise > 0:
            noise = np.random.normal(0, self.augment_noise, img_float.shape).astype(np.float32)
            img_float = np.clip(img_float + noise, 0.0, 1.0)

        return img_float

    def _open_h5(self):
        """Open HDF5 file lazily (on first access per call)."""
        if self.h5file is None:
            self.h5file = h5py.File(str(self.hdf5_path), 'r', rdcc_nbytes=1024*1024, rdcc_nslots=101)
        return self.h5file

    def _close_h5(self):
        """Close HDF5 file to free memory."""
        if self.h5file is not None:
            try:
                self.h5file.close()
            except:
                pass
            self.h5file = None

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample in LeRobot format with temporal observation stacking.

        Returns:
            Dictionary with LeRobot-compatible keys and values.
        """
        # Open file on demand and close after reading
        self._open_h5()
        # Calculate observation frame indices (temporal stacking)
        # For n_obs_steps=2: if idx=10, we get frames [9, 10]
        # For n_obs_steps=1: if idx=10, we get frame [10]
        obs_start_idx = max(0, idx - (self.n_obs_steps - 1))
        obs_indices = list(range(obs_start_idx, idx + 1))

        # Pad if we don't have enough history (at the beginning of episode)
        while len(obs_indices) < self.n_obs_steps:
            obs_indices.insert(0, obs_indices[0])  # Repeat first frame

        # Get phase information if available
        phase_id = None
        phase_label = "unknown"
        if self.has_phase:
            phase_id = int(self.h5file['phase'][idx])
            phase_label = self.phase_mapping.get(phase_id, "unknown")

        # Create LeRobot sample (metadata from current frame)
        lerobot_sample = {
            # Metadata
            "task": self.default_task_instruction,
            "phase": phase_label,
            "phase_id": phase_id if phase_id is not None else -1,
            "timestamp": float(self.h5file['timestamp'][idx]),
            "frame_index": idx,
            "episode_index": self.episode_index,
            "index": idx,  # Will be updated by ConcatDataset
        }

        # Determine which cameras to dropout (same for all temporal steps)
        active_cameras = self.selected_cameras.copy()
        if self.camera_dropout_prob > 0 and random.random() < self.camera_dropout_prob:
            # Calculate how many cameras to keep active
            num_to_keep = max(self.min_cameras, random.randint(self.min_cameras, len(self.selected_cameras)))
            if num_to_keep < len(self.selected_cameras):
                active_cameras = random.sample(self.selected_cameras, num_to_keep)

        # Process images: Load temporal observations and stack to (n_obs_steps, C, H, W)
        for cam_key in self.selected_cameras:
            actual_cam_name = self.camera_name_mapping[cam_key]
            try:
                # Check if this camera should be dropped out
                if cam_key not in active_cameras:
                    # Camera dropout: use black image (original size, float32 [0,1], C,H,W format)
                    # Get original image size from first frame
                    first_frame = self.h5file['observations']['images'][actual_cam_name][0]
                    if isinstance(first_frame, np.ndarray) and len(first_frame.shape) == 3:
                        h, w, c = first_frame.shape
                    else:
                        # JPEG format - use default size (will be decoded later)
                        h, w, c = 480, 640, 3  # Default assumption
                    lerobot_sample[f"observation.images.{cam_key}"] = torch.zeros(self.n_obs_steps, c, h, w, dtype=torch.float32)
                    continue

                # Load temporal images from HDF5 using actual camera name
                img_tensors = []
                for obs_idx in obs_indices:
                    raw_data = self.h5file['observations']['images'][actual_cam_name][obs_idx]

                    # Decode based on format
                    if self.is_jpeg_format:
                        # JPEG format: decode compressed binary
                        if isinstance(raw_data, np.ndarray):
                            jpeg_array = raw_data.flatten().astype(np.uint8)
                        elif isinstance(raw_data, (bytes, bytearray)):
                            jpeg_array = np.frombuffer(raw_data, dtype=np.uint8)
                        else:
                            jpeg_array = np.array(raw_data, dtype=np.uint8).flatten()

                        # Ensure contiguous array for cv2
                        if not jpeg_array.flags['C_CONTIGUOUS']:
                            jpeg_array = np.ascontiguousarray(jpeg_array)

                        # Decode JPEG: bytes → BGR → RGB
                        img_bgr = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
                        if img_bgr is None:
                            raise ValueError(f"Failed to decode JPEG data (size: {jpeg_array.size})")
                        img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    else:
                        # GZIP format: already decoded RGB array
                        img_np = raw_data

                    # Verify image shape
                    if len(img_np.shape) != 3 or img_np.shape[2] != 3:
                        raise ValueError(f"Invalid image shape: {img_np.shape}, expected (H, W, 3)")

                    # Apply augmentation (only to current frame, not to history)
                    # Augmentation is applied on uint8 images before conversion
                    if obs_idx == idx and self.augment:
                        # Convert to [0, 1] for augmentation
                        img_float = img_np.astype(np.float32) / 255.0
                        img_float = self.apply_image_augmentation(img_float)
                        # Convert back to uint8
                        img_np = (img_float * 255.0).astype(np.uint8)

                    # Keep as uint8 in (H, W, C) format - preprocessor will handle conversion
                    # The policy's preprocessor will:
                    # 1. Convert uint8 -> float32
                    # 2. Normalize to [0, 1]
                    # 3. Rearrange to (C, H, W) - PyTorch expects channels-first
                    # 4. SmolVLA will resize with padding to target size
                    img_tensor = torch.from_numpy(img_np).float() / 255.0  # (H,W,C) float32 [0,1]
                    img_tensor = img_tensor.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
                    img_tensors.append(img_tensor)

                    # FIXED: Explicit memory cleanup to prevent accumulation
                    del img_np
                    if self.is_jpeg_format and 'img_bgr' in locals():
                        del img_bgr
                    if 'img_float' in locals():
                        del img_float

                # Stack temporal observations: list of (C, H, W) -> (n_obs_steps, C, H, W)
                lerobot_sample[f"observation.images.{cam_key}"] = torch.stack(img_tensors, dim=0)

            except Exception as e:
                logger.warning(f"Failed to load {cam_key} ({actual_cam_name}) at frame {idx}: {e}")
                # Create dummy images if loading fails (original size, float32 [0,1], C,H,W format)
                first_frame = self.h5file['observations']['images'][actual_cam_name][0]
                if isinstance(first_frame, np.ndarray) and len(first_frame.shape) == 3:
                    h, w, c = first_frame.shape
                else:
                    h, w, c = 480, 640, 3
                lerobot_sample[f"observation.images.{cam_key}"] = torch.zeros(self.n_obs_steps, c, h, w, dtype=torch.float32)

        # Process robot state with temporal stacking
        # Stack states to shape (n_obs_steps, state_dim)
        state_tensors = []
        for obs_idx in obs_indices:
            state_parts = []

            if self.use_qpos:
                qpos = self.h5file['observations']['qpos'][obs_idx]
                state_parts.append(qpos)

            if self.use_ee_pose:
                ee_pose = self.h5file['observations']['ee_pose'][obs_idx]
                state_parts.append(ee_pose)

            # Concatenate robot state parts (normalized usually)
            if len(state_parts) > 1:
                robot_state = np.concatenate(state_parts, axis=0)
            elif len(state_parts) == 1:
                robot_state = state_parts[0]
            else:
                robot_state = np.array([], dtype=np.float32)

            # Convert to tensor
            state_tensor = torch.from_numpy(robot_state.astype(np.float32))

            # Apply state normalization only to robot state parts (before adding sensor)
            if self.normalizer is not None and state_tensor.numel() > 0:
                state_tensor = self.normalizer.normalize(
                    state_tensor, "observation.state"
                )

            # Process Sensor Data (append AFTER normalization)
            if self.use_sensor:
                # Load sensor distance (mm)
                # Logic: < 3.7mm -> 1.0 (detected), else -> 0.0 (not detected)
                if 'sensor_dist' in self.h5file['observations']:
                    raw_dist = self.h5file['observations']['sensor_dist'][obs_idx]
                    # Check if detected (>= 0) AND within range (< 3.7)
                    if 0.0 <= raw_dist < 3.7:
                        sensor_val = 1.0
                    else:
                        sensor_val = 0.0
                else:
                    # Fallback if sensor data missing
                    sensor_val = 0.0
                
                sensor_tensor = torch.tensor([sensor_val], dtype=torch.float32)
                
                # Concatenate: [Robot_State(Normalized), Sensor(0/1)]
                state_tensor = torch.cat([state_tensor, sensor_tensor], dim=0)

            state_tensors.append(state_tensor)

        # Stack temporal states: list of (state_dim,) -> (n_obs_steps, state_dim)
        lerobot_sample["observation.state"] = torch.stack(state_tensors, dim=0)

        # NOTE: Normalization is already applied per-step above to handle the sensor mismatch

        # Squeeze temporal dimension if requested (for ACT compatibility)
        if self.squeeze_n_obs_steps and self.n_obs_steps == 1:
            # Squeeze n_obs_steps dimension: (1, ...) -> (...)
            for key in list(lerobot_sample.keys()):
                if key.startswith("observation.images.") or key == "observation.state":
                    lerobot_sample[key] = lerobot_sample[key].squeeze(0)

        # Process actions
        if self.use_ee_pose_delta_as_action:
            # NEW LOGIC: Calculate action from ee_pose delta.
            if self.horizon == 1:
                # For a single step, action at `idx` is `pose[idx+1] - pose[idx]`.
                # __len__ ensures `idx+1` is always valid.
                pose_t0 = self.h5file['observations']['ee_pose'][idx]
                pose_t1 = self.h5file['observations']['ee_pose'][idx + 1]
                action = pose_t1 - pose_t0

                # Normalize orientation deltas to [-π, π] (handle angle wrapping)
                # Assumes action format: [x, y, z, rx, ry, rz]
                action[3:] = normalize_angle_delta(action[3:])

                lerobot_sample["action"] = torch.from_numpy(action.astype(np.float32))
                lerobot_sample["action_is_pad"] = torch.tensor([False], dtype=torch.bool)
            else:
                # For multi-step, we need poses from `idx` to `idx + horizon`.
                end_idx_for_poses = min(idx + self.horizon + 1, self.num_frames)
                poses = self.h5file['observations']['ee_pose'][idx:end_idx_for_poses]

                actions = poses[1:] - poses[:-1]

                # Normalize orientation deltas to [-π, π] for all steps
                for i in range(actions.shape[0]):
                    actions[i, 3:] = normalize_angle_delta(actions[i, 3:])

                original_len = actions.shape[0]

                # Pad if necessary
                if original_len < self.horizon:
                    if original_len > 0:
                        padding = np.repeat(actions[-1:], self.horizon - original_len, axis=0)
                    else:
                        padding = np.zeros((self.horizon, self.h5file['observations']['ee_pose'].shape[1]), dtype=np.float32)
                    actions = np.concatenate([actions, padding], axis=0)

                # Create action_is_pad mask
                action_is_pad = torch.zeros(self.horizon, dtype=torch.bool)
                if original_len < self.horizon:
                    action_is_pad[original_len:] = True

                lerobot_sample["action"] = torch.from_numpy(actions.astype(np.float32))
                lerobot_sample["action_is_pad"] = action_is_pad
        else:
            # ORIGINAL LOGIC: Use pre-recorded action from HDF5 file.
            if self.horizon == 1:
                # Single-step action
                action = self.h5file['action'][idx]
                lerobot_sample["action"] = torch.from_numpy(action.astype(np.float32))
                # No padding for single-step
                lerobot_sample["action_is_pad"] = torch.tensor([False], dtype=torch.bool)
            else:
                # Multi-step action chunk
                end_idx = min(idx + self.horizon, self.num_frames)
                actions = self.h5file['action'][idx:end_idx]
                original_len = actions.shape[0]

                # Pad if necessary
                if actions.shape[0] < self.horizon:
                    padding = np.repeat(actions[-1:], self.horizon - actions.shape[0], axis=0)
                    actions = np.concatenate([actions, padding], axis=0)

                # Create action_is_pad mask
                action_is_pad = torch.zeros(self.horizon, dtype=torch.bool)
                if original_len < self.horizon:
                    action_is_pad[original_len:] = True

                lerobot_sample["action"] = torch.from_numpy(actions.astype(np.float32))
                lerobot_sample["action_is_pad"] = action_is_pad

        # Apply action normalization if normalizer is available
        if self.normalizer is not None:
            lerobot_sample["action"] = self.normalizer.normalize(
                lerobot_sample["action"], "action"
            )

        # Determine which instruction to use for this frame
        current_task_str = self.default_task_instruction

        # Logic to determine phase:
        # 1. Use explicit 'phase' info if available
        # 2. Use 'sensor_dist' if available (Implicit phase inference)
        # 3. Fallback to default

        is_phase2 = False

        if self.has_phase:
            # If phase info exists: 1=Align (Phase1), 2=Insert (Phase2), 3=Hold (Phase2/3)
            p_id = int(self.h5file['phase'][idx])
            if p_id >= 2: # Phase 2 or 3 -> Insert instruction
                is_phase2 = True
        elif 'sensor_dist' in self.h5file['observations']:
            # Infer phase from sensor distance
            # Dist < 3.7mm means needle is close/inside -> Phase 2
            dist = self.h5file['observations']['sensor_dist'][idx]
            if dist < 3.7:
                is_phase2 = True

        # Select task string
        if is_phase2:
            current_task_str = self.task_instruction2
        else:
            current_task_str = self.task_instruction1

        # Update task string in metadata
        lerobot_sample["task"] = current_task_str

        # Add language tokens if tokenizer is available
        if self.tokenizer is not None:
            if is_phase2:
                selected_tokens = self.task2_tokens
                selected_mask = self.task2_mask
            else:
                selected_tokens = self.task1_tokens
                selected_mask = self.task1_mask

            lerobot_sample["observation.language.tokens"] = selected_tokens.clone()
            lerobot_sample["observation.language.attention_mask"] = selected_mask.clone()

        # Close HDF5 file after reading to prevent memory accumulation
        # with 30k+ episodes, keeping all files open exhausts system RAM
        self._close_h5()

        return lerobot_sample

    def __getstate__(self):
        """Close HDF5 file before pickling for multiprocessing."""
        state = self.__dict__.copy()
        # h5file is opened lazily, ensure it's closed before pickling
        if 'h5file' in state and state['h5file'] is not None:
            try:
                state['h5file'].close()
            except:
                pass
            state['h5file'] = None
        return state

    def __setstate__(self, state):
        """Restore state after unpickling. HDF5 file will be opened lazily."""
        self.__dict__.update(state)
        self.h5file = None  # Will be opened on-demand in __getitem__

    def __del__(self):
        """Close HDF5 file when dataset is destroyed."""
        self._close_h5()


def create_hdf5_lerobot_dataset(
    hdf5_paths: List[Union[str, Path]],
    horizon: int = 1,
    n_obs_steps: int = 1,
    use_qpos: bool = False,
    use_ee_pose: bool = True,
    use_sensor: bool = False,
    use_ee_pose_delta_as_action: bool = False,
    task_instruction: str = "Move the needle to the eye phantom and insert it through the trocar opening",
    task_instruction1: str = "Move the needle to the eye phantom",
    task_instruction2: str = "insert the needle through the eye phantom trocar opening",
    cameras: Optional[List[str]] = None,
    camera_dropout_prob: float = 0.0,
    min_cameras: int = 1,
    augment: bool = True,
    augment_brightness: float = 0.2,
    augment_contrast: float = 0.2,
    augment_saturation: float = 0.2,
    augment_hue: float = 0.05,
    augment_noise: float = 0.02,
    squeeze_n_obs_steps: bool = False,
    tokenizer: Optional[Any] = None,
    tokenizer_max_length: int = 48,
    normalizer: Optional[Any] = None,
) -> Dataset:
    """
    Create a combined HDF5 LeRobot dataset from multiple episodes.

    Args:
        hdf5_paths: List of HDF5 file paths
        horizon: Action prediction horizon
        n_obs_steps: Number of observation steps (temporal stacking)
        use_qpos: Use joint positions for state
        use_ee_pose: Use end-effector pose for state (default)
        use_ee_pose_delta_as_action: Calculate action from ee_pose delta
        task_instruction: Task instruction text
        cameras: List of camera names to use (e.g., ["camera1", "camera3"]). None = use all cameras
        camera_dropout_prob: Probability of dropping out cameras
        min_cameras: Minimum number of cameras to keep active
        augment: Enable image augmentation
        augment_brightness: Max brightness adjustment factor
        augment_contrast: Max contrast adjustment factor
        augment_saturation: Max saturation adjustment factor
        augment_hue: Max hue adjustment (in degrees / 360)
        augment_noise: Gaussian noise std deviation
        squeeze_n_obs_steps: Squeeze temporal dimension if n_obs_steps=1 (for ACT)
        tokenizer: HuggingFace tokenizer for language conditioning (optional)
        tokenizer_max_length: Maximum length for tokenized text
        normalizer: Normalizer instance for action/state normalization (optional)

    Returns:
        Combined dataset (ConcatDataset if multiple episodes)
    """
    datasets = []

    for episode_idx, hdf5_path in enumerate(hdf5_paths):
        file_path = Path(hdf5_path)

        # Check if file exists
        if not file_path.exists():
            logger.warning(f"HDF5 file does not exist: {file_path}")
            continue

        # Create dataset for this episode
        try:
            dataset = HDF5LeRobotDataset(
                hdf5_path=str(file_path),
                episode_index=episode_idx,
                horizon=horizon,
                n_obs_steps=n_obs_steps,
                use_qpos=use_qpos,
                use_ee_pose=use_ee_pose,
                use_sensor=use_sensor,
                use_ee_pose_delta_as_action=use_ee_pose_delta_as_action,
                task_instruction=task_instruction,
                task_instruction1=task_instruction1,
                task_instruction2=task_instruction2,
                cameras=cameras,
                camera_dropout_prob=camera_dropout_prob,
                min_cameras=min_cameras,
                augment=augment,
                augment_brightness=augment_brightness,
                augment_contrast=augment_contrast,
                augment_saturation=augment_saturation,
                augment_hue=augment_hue,
                augment_noise=augment_noise,
                squeeze_n_obs_steps=squeeze_n_obs_steps,
                tokenizer=tokenizer,
                tokenizer_max_length=tokenizer_max_length,
                normalizer=normalizer,
            )
            datasets.append(dataset)
        except Exception as e:
            logger.error(f"Failed to load episode {file_path}: {e}")
            continue

    if len(datasets) == 0:
        raise ValueError("No valid episodes found!")

    logger.info(f"Created combined dataset with {len(datasets)} episodes")

    # Combine all episodes
    if len(datasets) == 1:
        combined_dataset = datasets[0]
    else:
        combined_dataset = ConcatDataset(datasets)

    logger.info(f"Total samples: {len(combined_dataset)}")

    return combined_dataset


def hdf5_lerobot_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for HDF5 LeRobot batches.

    Handles:
    - Stacking tensors (images, state, actions)
    - Keeping lists (task strings)
    - Maintaining metadata

    Args:
        batch: List of samples from HDF5LeRobotDataset

    Returns:
        Batched dictionary
    """
    if len(batch) == 0:
        return {}

    # Get all keys from first sample
    keys = batch[0].keys()

    batched = {}

    for key in keys:
        values = [sample[key] for sample in batch]

        # Handle different types
        if key in ["task", "phase"]:
            # Keep as list of strings
            batched[key] = values
        elif key in ["timestamp", "frame_index", "episode_index", "index", "phase_id"]:
            # Stack scalars into tensor
            batched[key] = torch.tensor(values)
        elif isinstance(values[0], torch.Tensor):
            # Stack tensors
            batched[key] = torch.stack(values, dim=0)
        else:
            # Keep as list for other types
            batched[key] = values

    return batched


if __name__ == "__main__":
    """
    Test the HDF5 LeRobot adapter.
    """
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🧪 Testing HDF5 LeRobot Adapter...\n")

    # Test with a single episode
    test_file = "/home/najo/NAS/VLA/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar/260106/2_JYT/episode_20260106_134625_trimmed_0_556.h5"

    try:
        dataset = HDF5LeRobotDataset(
            hdf5_path=test_file,
            episode_index=0,
            use_ee_pose=True,
            use_qpos=False,
        )

        print(f"✅ Dataset created: {len(dataset)} samples\n")

        # Test first sample
        sample = dataset[0]

        print("📊 Sample structure:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {type(value).__name__} = {value if not isinstance(value, str) else value[:50]}")

        print("\n✅ Sample loaded successfully!\n")

        # Test dataloader
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=hdf5_lerobot_collate_fn,
            num_workers=0,
        )

        batch = next(iter(dataloader))

        print("📦 Batch structure:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            elif isinstance(value, list):
                print(f"  {key}: List[{type(value[0]).__name__}] (len={len(value)})")

        print("\n✅ DataLoader works!\n")

        print("\n🎉 All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
