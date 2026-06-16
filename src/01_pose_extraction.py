# ============================================================
# Violence Detection — Step 1: Pose Extraction
# Nejra Gutic & Zeynep Nur Yilmaz, ITU 2026
#
# What this script does:
#   - Loads RWF-2000 videos from DATA_ROOT
#   - Runs YOLO11n-pose on every frame of every video
#   - Selects the top-2 most salient persons per frame
#     (salience = bounding_box_area x mean_keypoint_confidence)
#   - Saves keypoints as .npy files with shape (T, 2, 17, 3)
#     where T=frames, 2=persons, 17=COCO keypoints, 3=(x,y,conf)
#
# Output: OUT_ROOT/{split}/{class}/{video_name}.npy
# ============================================================

import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from ultralytics import YOLO

# ── Paths — update these to match your local setup ───────────
DATA_ROOT = "data/rwf2000_clean"       # input: RGB videos
OUT_ROOT  = "data/processed/pose/npy"  # output: .npy keypoints

SPLITS  = ["train", "val"]
CLASSES = ["Fight", "NonFight"]

# ── YOLO settings ─────────────────────────────────────────────
MODEL_NAME   = "yolo11n-pose.pt"
CONF_THRESH  = 0.25
IMG_SIZE     = 640
FRAME_STRIDE = 1  # process every frame


def choose_top2_persons(boxes_xyxy, keypoints_conf, max_persons=2):
    """
    Select the top-N most salient persons from a frame.
    Salience score = bounding_box_area x mean_keypoint_confidence.
    This favors persons who are both large in frame AND clearly detected.
    Returns indices of top persons sorted by salience (highest first).
    """
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return []

    scores = []
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        conf = float(np.mean(keypoints_conf[i])) if keypoints_conf is not None and len(keypoints_conf) > i else 1.0
        scores.append((area * conf, i))

    scores.sort(reverse=True)
    return [idx for _, idx in scores[:max_persons]]


def extract_video_keypoints(video_path, model, conf=0.25, imgsz=640,
                             max_frames=None, frame_stride=1):
    """
    Extract YOLO11n-pose keypoints from every frame of a video.

    Returns:
        keypoints: np.ndarray of shape (T, 2, 17, 3)
                   T=frames, 2=persons, 17=COCO keypoints, 3=(x,y,confidence)
        fps: float
        total_frames_read: int
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    saved = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        results = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)

        frame_kpts = np.zeros((2, 17, 3), dtype=np.float32)

        if len(results) > 0:
            r = results[0]
            has_boxes = hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0
            has_kpts  = hasattr(r, "keypoints") and r.keypoints is not None

            if has_boxes and has_kpts:
                boxes_xyxy     = r.boxes.xyxy.cpu().numpy()
                keypoints_xy   = r.keypoints.xy.cpu().numpy()
                keypoints_conf = r.keypoints.conf.cpu().numpy() if getattr(r.keypoints, "conf", None) is not None else None

                top_idxs = choose_top2_persons(boxes_xyxy, keypoints_conf)

                for person_slot, idx in enumerate(top_idxs):
                    frame_kpts[person_slot, :, :2] = keypoints_xy[idx][:17]
                    frame_kpts[person_slot, :, 2]  = keypoints_conf[idx][:17] if keypoints_conf is not None else 1.0

        saved.append(frame_kpts)
        frame_idx += 1

        if max_frames is not None and len(saved) >= max_frames:
            break

    cap.release()

    if len(saved) == 0:
        return np.zeros((0, 2, 17, 3), dtype=np.float32), fps, frame_idx

    return np.stack(saved, axis=0).astype(np.float32), fps, frame_idx


def process_split_class(split, cls, model, conf=0.25, imgsz=640,
                         frame_stride=1, overwrite=False):
    in_dir  = os.path.join(DATA_ROOT, split, cls)
    out_dir = os.path.join(OUT_ROOT, split, cls)
    os.makedirs(out_dir, exist_ok=True)

    video_paths = sorted(
        glob(os.path.join(in_dir, "*.avi")) +
        glob(os.path.join(in_dir, "*.mp4"))
    )

    print(f"{split}/{cls}: {len(video_paths)} videos")
    failed = []

    for vp in tqdm(video_paths, desc=f"{split}-{cls}"):
        base     = os.path.splitext(os.path.basename(vp))[0]
        out_path = os.path.join(out_dir, base + ".npy")

        if os.path.exists(out_path) and not overwrite:
            continue

        try:
            kpts, fps, nread = extract_video_keypoints(
                video_path=vp, model=model,
                conf=conf, imgsz=imgsz,
                max_frames=None, frame_stride=frame_stride
            )
            np.save(out_path, kpts)
        except Exception as e:
            failed.append((vp, str(e)))

    return failed


if __name__ == "__main__":
    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUT_ROOT, split, cls), exist_ok=True)

    model = YOLO(MODEL_NAME)
    print(f"Loaded: {MODEL_NAME}")

    all_failed = []
    for split in SPLITS:
        for cls in CLASSES:
            failed = process_split_class(
                split=split, cls=cls, model=model,
                conf=CONF_THRESH, imgsz=IMG_SIZE,
                frame_stride=FRAME_STRIDE, overwrite=False
            )
            all_failed.extend(failed)

    print(f"\nDone. Failed videos: {len(all_failed)}")
    for path, err in all_failed:
        print(f"  {path}: {err}")
