# ============================================================
# Violence Detection — Step 2: Pose Preprocessing
# Nejra Gutic & Zeynep Nur Yilmaz, ITU 2026
#
# What this script does:
#   - Loads .npy keypoint files produced by 01_pose_extraction.py
#   - Converts shape from (T, 2, 17, 3) to MMAction2 format:
#       keypoint:       (2, T, 17, 2)  — x,y coordinates
#       keypoint_score: (2, T, 17)     — confidence values
#   - Builds one big annotation .pkl file for the whole dataset
#   - The .pkl contains train/val split info + all annotations
#
# Output: OUT_DIR/rwf_pose.pkl
# ============================================================

import os
import glob
import numpy as np
import pickle
from tqdm import tqdm

# ── Paths — update these to match your local setup ───────────
POSE_ROOT = "data/processed/pose/npy"  # input: .npy files from step 1
OUT_DIR   = "data/processed/pose/pkl"  # output: .pkl annotation file

SPLITS    = ["train", "val"]
CLASSES   = ["Fight", "NonFight"]

# Label mapping: NonFight=0, Fight=1
LABEL_MAP = {"NonFight": 0, "Fight": 1}


def build_annotations():
    """
    Read all .npy files and build MMAction2-compatible annotation list.

    Each annotation is a dict with:
        frame_dir:      str  — identifier like 'train/Fight/000001'
        label:          int  — 0=NonFight, 1=Fight
        total_frames:   int  — T (number of frames in video)
        keypoint:       np.ndarray (2, T, 17, 2) — x,y coordinates
        keypoint_score: np.ndarray (2, T, 17)    — confidence scores
    """
    annotations = []
    split_dict  = {"train": [], "val": []}
    bad_files   = []

    for split_name in SPLITS:
        for class_name in CLASSES:
            folder    = os.path.join(POSE_ROOT, split_name, class_name)
            npy_files = sorted(glob.glob(os.path.join(folder, "*.npy")))

            print(f"\nProcessing {split_name}/{class_name} -> {len(npy_files)} files")

            for npy_path in tqdm(npy_files):
                video_name = os.path.splitext(os.path.basename(npy_path))[0]
                frame_dir  = f"{split_name}/{class_name}/{video_name}"

                try:
                    arr = np.load(npy_path)
                except Exception as e:
                    print(f"\nSkipping bad file: {npy_path} — {e}")
                    bad_files.append((npy_path, str(e)))
                    continue

                # Validate shape: must be (T, 2, 17, 3)
                if arr.ndim != 4 or arr.shape[1:] != (2, 17, 3):
                    print(f"\nSkipping unexpected shape: {npy_path} -> {arr.shape}")
                    bad_files.append((npy_path, f"unexpected shape {arr.shape}"))
                    continue

                T = arr.shape[0]

                # Split into coordinates and confidence
                # (T, 2, 17, 3) -> coords: (T, 2, 17, 2), scores: (T, 2, 17)
                coords = arr[..., :2]
                scores = arr[..., 2]

                # Transpose to MMAction2 format
                # coords: (T, 2, 17, 2) -> (2, T, 17, 2)
                # scores: (T, 2, 17)    -> (2, T, 17)
                keypoint       = np.transpose(coords, (1, 0, 2, 3)).astype(np.float32)
                keypoint_score = np.transpose(scores, (1, 0, 2)).astype(np.float32)

                annotations.append({
                    "frame_dir":      frame_dir,
                    "label":          LABEL_MAP[class_name],
                    "total_frames":   T,
                    "keypoint":       keypoint,
                    "keypoint_score": keypoint_score
                })
                split_dict[split_name].append(frame_dir)

    return annotations, split_dict, bad_files


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Building annotations...")
    annotations, split_dict, bad_files = build_annotations()

    print(f"\nTotal annotations : {len(annotations)}")
    print(f"Train samples     : {len(split_dict['train'])}")
    print(f"Val samples       : {len(split_dict['val'])}")
    print(f"Bad files         : {len(bad_files)}")

    overlap = set(split_dict["train"]) & set(split_dict["val"])
    print(f"Train/Val overlap : {len(overlap)} (should be 0)")

    # Save .pkl file
    ann_file = os.path.join(OUT_DIR, "rwf_pose.pkl")
    data = {"split": split_dict, "annotations": annotations}

    with open(ann_file, "wb") as f:
        pickle.dump(data, f)

    print(f"\nSaved: {ann_file}")

    # Verify
    with open(ann_file, "rb") as f:
        verify = pickle.load(f)

    sample = verify["annotations"][0]
    print(f"\nVerification — first sample:")
    print(f"  frame_dir      : {sample['frame_dir']}")
    print(f"  label          : {sample['label']}")
    print(f"  total_frames   : {sample['total_frames']}")
    print(f"  keypoint shape : {sample['keypoint'].shape}")
    print(f"  score shape    : {sample['keypoint_score'].shape}")
