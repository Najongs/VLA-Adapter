"""
convert_h5_to_rlds.py

Converts custom HDF5 episodes (needle insertion sim) to RLDS/TFDS format
compatible with VLA-Adapter training.

Usage:
    python custom_dataset/convert_h5_to_rlds.py \
        --input_dir custom_dataset/New2/collected_data_merged \
        --output_dir /data/public/NAS/LIBERO_modified/needle_insertion \
        --dataset_name needle_insertion \
        --max_episodes 0
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def decode_jpeg(jpeg_bytes):
    """Decode JPEG bytes stored in HDF5 vlen dataset to numpy array."""
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_image(img, target_size=(256, 256)):
    """Resize image to target size."""
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)


def process_episode(h5_path, target_img_size=(256, 256)):
    """
    Process a single H5 episode file and return RLDS-compatible steps.

    H5 structure:
        action: (N, 7) - delta_ee(6) + gripper(1), gripper: -1=open, 1=closed
        observations/ee_pose: (N, 7) - pos_mm(3) + euler_rad(3) + gripper(1)
        observations/qpos: (N, 6) - joint angles in degrees
        observations/images/side_camera: (N,) JPEG encoded
        observations/images/tool_camera: (N,) JPEG encoded
        language_instruction: string
        phase: (N,) int32

    RLDS output per step:
        action: (7,) float32 - delta_ee(6) + gripper(1)
        observation/image: (256, 256, 3) uint8 - side_camera (primary)
        observation/wrist_image: (256, 256, 3) uint8 - tool_camera (wrist)
        observation/state: (8,) float32 - EEF pos_mm(3) + euler_rad(3) + gripper(1) + phase(1)
        observation/joint_state: (6,) float32 - joint angles (degrees)
        language_instruction: string
        is_first, is_last, is_terminal: bool
        reward, discount: float32
    """
    with h5py.File(h5_path, "r") as f:
        actions = f["action"][:]  # (N, 7)
        ee_pose = f["observations/ee_pose"][:]  # (N, 7): pos_mm(3) + euler(3) + gripper(1)
        qpos = f["observations/qpos"][:]  # (N, 6)
        phase = f["phase"][:]  # (N,)

        # Language instruction
        lang_raw = f["language_instruction"][()]
        if isinstance(lang_raw, bytes):
            lang = lang_raw.decode("utf-8")
        else:
            lang = str(lang_raw)

        n_steps = actions.shape[0]

        # Build state: ee_pos_mm(3) + euler_rad(3) + gripper(1) + gripper_duplicate(1) = 8 dim
        # We duplicate gripper to match LIBERO's 8-dim state convention (6D EEF + 2D gripper)
        state = np.concatenate([
            ee_pose[:, :6],           # EEF state: pos_mm(3) + euler_rad(3)
            ee_pose[:, 6:7],          # gripper state 1
            ee_pose[:, 6:7],          # gripper state 2 (duplicate for compatibility)
        ], axis=-1).astype(np.float32)  # (N, 8)

        steps = []
        for i in range(n_steps):
            # Decode and resize images
            side_jpeg = f["observations/images/side_camera"][i]
            tool_jpeg = f["observations/images/tool_camera"][i]
            top_jpeg = f["observations/images/top_camera"][i]

            side_img = resize_image(decode_jpeg(side_jpeg), target_img_size)
            tool_img = resize_image(decode_jpeg(tool_jpeg), target_img_size)
            top_img = resize_image(decode_jpeg(top_jpeg), target_img_size)

            step = {
                "action": actions[i].astype(np.float32),
                "observation": {
                    "image": side_img,
                    "wrist_image": tool_img,
                    "top_image": top_img,
                    "state": state[i],
                    "joint_state": qpos[i].astype(np.float32),
                },
                "language_instruction": lang,
                "is_first": i == 0,
                "is_last": i == n_steps - 1,
                "is_terminal": i == n_steps - 1,
                "reward": 1.0 if i == n_steps - 1 else 0.0,
                "discount": 1.0,
            }
            steps.append(step)

    return steps


def steps_to_tf_example(steps, episode_path):
    """Convert a list of steps to a tf.train.SequenceExample."""
    # Build features for each step
    step_features = {
        "action": [],
        "observation/image": [],
        "observation/wrist_image": [],
        "observation/state": [],
        "observation/joint_state": [],
        "language_instruction": [],
        "is_first": [],
        "is_last": [],
        "is_terminal": [],
        "reward": [],
        "discount": [],
    }

    for step in steps:
        step_features["action"].append(step["action"])
        step_features["observation/state"].append(step["observation"]["state"])
        step_features["observation/joint_state"].append(step["observation"]["joint_state"])
        step_features["language_instruction"].append(step["language_instruction"])
        step_features["is_first"].append(step["is_first"])
        step_features["is_last"].append(step["is_last"])
        step_features["is_terminal"].append(step["is_terminal"])
        step_features["reward"].append(step["reward"])
        step_features["discount"].append(step["discount"])

        # Encode images as JPEG
        for key in ["image", "wrist_image"]:
            img = step["observation"][key]
            _, encoded = cv2.imencode(".jpeg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            step_features[f"observation/{key}"].append(encoded.tobytes())

    return step_features


def write_tfrecord(episodes_data, output_path):
    """Write episodes to a TFRecord file."""
    writer = tf.io.TFRecordWriter(str(output_path))

    for ep_steps, ep_path in episodes_data:
        # Create tf.train.Example for each episode
        steps_dict = {
            "steps": {
                "action": np.array([s["action"] for s in ep_steps], dtype=np.float32),
                "is_first": np.array([s["is_first"] for s in ep_steps], dtype=bool),
                "is_last": np.array([s["is_last"] for s in ep_steps], dtype=bool),
                "is_terminal": np.array([s["is_terminal"] for s in ep_steps], dtype=bool),
                "reward": np.array([s["reward"] for s in ep_steps], dtype=np.float32),
                "discount": np.array([s["discount"] for s in ep_steps], dtype=np.float32),
                "language_instruction": [s["language_instruction"] for s in ep_steps],
                "observation": {
                    "image": [s["observation"]["image"] for s in ep_steps],
                    "wrist_image": [s["observation"]["wrist_image"] for s in ep_steps],
                    "state": np.array([s["observation"]["state"] for s in ep_steps], dtype=np.float32),
                    "joint_state": np.array([s["observation"]["joint_state"] for s in ep_steps], dtype=np.float32),
                },
            },
            "episode_metadata": {
                "file_path": str(ep_path),
            },
        }

        # Serialize using tf.train.Example
        serialized = serialize_episode(steps_dict)
        writer.write(serialized)

    writer.close()


def serialize_episode(episode_dict):
    """Serialize an episode dict to tf.train.Example bytes using TFDS-compatible format."""
    import tensorflow_datasets as tfds

    steps = episode_dict["steps"]
    n_steps = len(steps["action"])

    # Build individual step examples
    step_records = []
    for i in range(n_steps):
        # Encode image as JPEG bytes
        img = steps["observation"]["image"][i]
        _, img_encoded = cv2.imencode(".jpeg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        wrist_img = steps["observation"]["wrist_image"][i]
        _, wrist_encoded = cv2.imencode(".jpeg", cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))

        step_record = {
            "action": steps["action"][i],
            "observation": {
                "image": img_encoded.tobytes(),
                "wrist_image": wrist_encoded.tobytes(),
                "state": steps["observation"]["state"][i],
                "joint_state": steps["observation"]["joint_state"][i],
            },
            "language_instruction": steps["language_instruction"][i],
            "is_first": steps["is_first"][i],
            "is_last": steps["is_last"][i],
            "is_terminal": steps["is_terminal"][i],
            "reward": steps["reward"][i],
            "discount": steps["discount"][i],
        }
        step_records.append(step_record)

    return step_records


def build_tfds_dataset(input_dir, output_dir, dataset_name, max_episodes=0,
                       target_img_size=(256, 256), num_shards=16):
    """
    Build a TFDS-compatible dataset from H5 files using tensorflow_datasets builder API.
    """
    import tensorflow_datasets as tfds

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    version_dir = output_path / "1.0.0"
    version_dir.mkdir(parents=True, exist_ok=True)

    # Collect all h5 files
    h5_files = sorted(input_path.glob("*.h5"))
    if max_episodes > 0:
        h5_files = h5_files[:max_episodes]

    print(f"Found {len(h5_files)} episodes")

    # Determine number of shards
    actual_shards = min(num_shards, len(h5_files))
    episodes_per_shard = (len(h5_files) + actual_shards - 1) // actual_shards

    shard_lengths = []

    for shard_idx in range(actual_shards):
        start = shard_idx * episodes_per_shard
        end = min(start + episodes_per_shard, len(h5_files))
        shard_files = h5_files[start:end]

        if not shard_files:
            break

        tfrecord_path = version_dir / f"{dataset_name}-train.tfrecord-{shard_idx:05d}-of-{actual_shards:05d}"

        writer = tf.io.TFRecordWriter(str(tfrecord_path))
        shard_count = 0

        for h5_file in tqdm(shard_files, desc=f"Shard {shard_idx}/{actual_shards}"):
            try:
                steps = process_episode(h5_file, target_img_size)
            except Exception as e:
                print(f"Error processing {h5_file}: {e}")
                continue

            # Serialize episode as TFDS expects
            serialized = serialize_tfds_episode(steps, str(h5_file))
            writer.write(serialized)
            shard_count += 1

        writer.close()
        shard_lengths.append(str(shard_count))

    # Write dataset_info.json
    total_episodes = sum(int(s) for s in shard_lengths)
    total_bytes = sum(
        os.path.getsize(version_dir / f"{dataset_name}-train.tfrecord-{i:05d}-of-{actual_shards:05d}")
        for i in range(len(shard_lengths))
    )

    dataset_info = {
        "citation": "",
        "description": "Custom needle insertion simulation dataset for VLA-Adapter training.",
        "fileFormat": "tfrecord",
        "moduleName": f"{dataset_name}.{dataset_name}_dataset_builder",
        "name": dataset_name,
        "releaseNotes": {"1.0.0": "Initial release."},
        "splits": [
            {
                "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
                "name": "train",
                "numBytes": str(total_bytes),
                "shardLengths": shard_lengths,
            }
        ],
        "version": "1.0.0",
    }

    with open(version_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    # Write features.json
    features = {
        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
        "featuresDict": {
            "features": {
                "steps": {
                    "pythonClassName": "tensorflow_datasets.core.features.dataset_feature.Dataset",
                    "sequence": {
                        "feature": {
                            "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                            "featuresDict": {
                                "features": {
                                    "action": {
                                        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                        "tensor": {
                                            "shape": {"dimensions": ["7"]},
                                            "dtype": "float32",
                                            "encoding": "none",
                                        },
                                        "description": "Robot EEF delta action (6D) + gripper (1D).",
                                    },
                                    "is_terminal": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                                        "description": "True on last step.",
                                    },
                                    "is_last": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                                        "description": "True on last step.",
                                    },
                                    "language_instruction": {
                                        "pythonClassName": "tensorflow_datasets.core.features.text_feature.Text",
                                        "text": {},
                                        "description": "Language instruction.",
                                    },
                                    "observation": {
                                        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                                        "featuresDict": {
                                            "features": {
                                                "image": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
                                                    "image": {
                                                        "shape": {"dimensions": ["256", "256", "3"]},
                                                        "dtype": "uint8",
                                                        "encodingFormat": "jpeg",
                                                    },
                                                    "description": "Side camera RGB observation.",
                                                },
                                                "wrist_image": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
                                                    "image": {
                                                        "shape": {"dimensions": ["256", "256", "3"]},
                                                        "dtype": "uint8",
                                                        "encodingFormat": "jpeg",
                                                    },
                                                    "description": "Tool camera RGB observation.",
                                                },
                                                "top_image": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
                                                    "image": {
                                                        "shape": {"dimensions": ["256", "256", "3"]},
                                                        "dtype": "uint8",
                                                        "encodingFormat": "jpeg",
                                                    },
                                                    "description": "Top camera RGB observation.",
                                                },
                                                "state": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {
                                                        "shape": {"dimensions": ["8"]},
                                                        "dtype": "float32",
                                                        "encoding": "none",
                                                    },
                                                    "description": "Robot EEF state (6D pose + 2D gripper).",
                                                },
                                                "joint_state": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {
                                                        "shape": {"dimensions": ["6"]},
                                                        "dtype": "float32",
                                                        "encoding": "none",
                                                    },
                                                    "description": "Robot joint angles (degrees).",
                                                },
                                            }
                                        },
                                    },
                                    "is_first": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                                        "description": "True on first step.",
                                    },
                                    "discount": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "float32", "encoding": "none"},
                                        "description": "Discount, default 1.",
                                    },
                                    "reward": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "float32", "encoding": "none"},
                                        "description": "Reward, 1 on final step.",
                                    },
                                }
                            },
                        },
                        "length": "-1",
                    },
                },
                "episode_metadata": {
                    "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                    "featuresDict": {
                        "features": {
                            "file_path": {
                                "pythonClassName": "tensorflow_datasets.core.features.text_feature.Text",
                                "text": {},
                                "description": "Path to the original data file.",
                            }
                        }
                    },
                },
            }
        },
    }

    with open(version_dir / "features.json", "w") as f:
        json.dump(features, f, indent=2)

    print(f"\nDataset created at: {output_path}")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Shards: {len(shard_lengths)}")
    print(f"  Total bytes: {total_bytes}")


def _get_tfds_features():
    """Build TFDS features spec (cached)."""
    import tensorflow_datasets as tfds
    return tfds.features.FeaturesDict({
        "steps": tfds.features.Dataset({
            "action": tfds.features.Tensor(shape=(7,), dtype=tf.float32),
            "observation": tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=(256, 256, 3), encoding_format="jpeg"),
                "wrist_image": tfds.features.Image(shape=(256, 256, 3), encoding_format="jpeg"),
                "top_image": tfds.features.Image(shape=(256, 256, 3), encoding_format="jpeg"),
                "state": tfds.features.Tensor(shape=(8,), dtype=tf.float32),
                "joint_state": tfds.features.Tensor(shape=(6,), dtype=tf.float32),
            }),
            "language_instruction": tfds.features.Text(),
            "is_first": tfds.features.Scalar(dtype=tf.bool),
            "is_last": tfds.features.Scalar(dtype=tf.bool),
            "is_terminal": tfds.features.Scalar(dtype=tf.bool),
            "reward": tfds.features.Scalar(dtype=tf.float32),
            "discount": tfds.features.Scalar(dtype=tf.float32),
        }),
        "episode_metadata": tfds.features.FeaturesDict({
            "file_path": tfds.features.Text(),
        }),
    })


_FEATURES = None
_SERIALIZER = None


def serialize_tfds_episode(steps, file_path):
    """Serialize episode steps to bytes using TFDS ExampleSerializer."""
    import tensorflow_datasets as tfds

    global _FEATURES, _SERIALIZER
    if _FEATURES is None:
        _FEATURES = _get_tfds_features()
        _SERIALIZER = tfds.core.example_serializer.ExampleSerializer(
            _FEATURES.get_serialized_info()
        )

    episode = {
        "steps": [
            {
                "action": step["action"],
                "observation": {
                    "image": step["observation"]["image"],
                    "wrist_image": step["observation"]["wrist_image"],
                    "top_image": step["observation"]["top_image"],
                    "state": step["observation"]["state"],
                    "joint_state": step["observation"]["joint_state"],
                },
                "language_instruction": step["language_instruction"],
                "is_first": step["is_first"],
                "is_last": step["is_last"],
                "is_terminal": step["is_terminal"],
                "reward": step["reward"],
                "discount": step["discount"],
            }
            for step in steps
        ],
        "episode_metadata": {
            "file_path": file_path,
        },
    }

    encoded = _FEATURES.encode_example(episode)
    return _SERIALIZER.serialize_example(encoded)


def main():
    parser = argparse.ArgumentParser(description="Convert H5 episodes to RLDS/TFDS format")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing H5 episode files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for RLDS dataset")
    parser.add_argument("--dataset_name", type=str, default="needle_insertion",
                        help="Dataset name for RLDS")
    parser.add_argument("--max_episodes", type=int, default=0,
                        help="Max episodes to convert (0 = all)")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Target image size (square)")
    parser.add_argument("--num_shards", type=int, default=16,
                        help="Number of TFRecord shards")
    args = parser.parse_args()

    build_tfds_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        max_episodes=args.max_episodes,
        target_img_size=(args.img_size, args.img_size),
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
