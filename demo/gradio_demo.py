# ============================================================
# Violence Detection — Gradio Demo
# Nejra Gutic & Zeynep Nur Yilmaz, ITU 2026
#
# What this script does:
#   - Loads all three trained models:
#       RGB stream:   R3D-18 (checkpoints/rgb_v2/best_model.pth)
#       Joint stream: 2s-AGCN (checkpoints/pose/2s-agcn-joint/...)
#       Bone stream:  2s-AGCN (checkpoints/pose/2s-agcn_bone/...)
#   - Launches a Gradio web interface where you can upload any video
#   - Runs the full three-stream late fusion pipeline on the video
#   - Outputs: Fight/NonFight prediction + confidence + per-stream scores
#
# Fusion weights (grid search optimal):
#   RGB=0.45, Joint=0.25, Bone=0.30
#
# Requirements:
#   - MMAction2 installed
#   - All three model checkpoints available
#   - pip install gradio ultralytics decord
#
# Usage:
#   python gradio_demo.py
#   Then open the local URL shown in terminal
# ============================================================

import os
import sys
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import gradio as gr
from torchvision.models.video import r3d_18
from decord import VideoReader, cpu
from ultralytics import YOLO

# ── Paths — update these to match your local setup ───────────
BASE        = "."
RGB_CKPT    = f"{BASE}/checkpoints/rgb_v2/best_model.pth"
JOINT_CKPT  = f"{BASE}/checkpoints/pose/2s-agcn-joint/best_acc_top1_epoch_8.pth"
BONE_CKPT   = f"{BASE}/checkpoints/pose/2s-agcn_bone/best_acc_top1_epoch_8.pth"
JOINT_CFG   = f"{BASE}/configs/2s-agcn_rwf2000_joint.py"
BONE_CFG    = f"{BASE}/configs/2s-agcn_rwf2000_bone.py"
MMACTION2   = "/path/to/mmaction2"   # update this

# ── Fusion weights ────────────────────────────────────────────
W_RGB, W_JOINT, W_BONE = 0.45, 0.25, 0.30
CLASS_NAMES = ["NonFight", "Fight"]

# ── COCO skeleton parent indices (for bone vector computation) ─
COCO_PARENTS = [0, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Load models ───────────────────────────────────────────────
def load_models():
    sys.path.insert(0, MMACTION2)
    from mmaction.apis import init_recognizer
    from mmaction.utils import register_all_modules
    register_all_modules()

    # RGB model
    rgb_model = r3d_18(weights=None)
    rgb_model.fc = nn.Linear(rgb_model.fc.in_features, 2)
    state = torch.load(RGB_CKPT, map_location=device, weights_only=False)
    state_dict = state.get("model_state_dict", state.get("state_dict", state))
    rgb_model.load_state_dict(state_dict)
    rgb_model = rgb_model.to(device).eval()
    print("✅ RGB model loaded")

    # Joint model
    joint_model = init_recognizer(JOINT_CFG, JOINT_CKPT, device=str(device))
    joint_model.eval()
    print("✅ Joint model loaded")

    # Bone model
    bone_model = init_recognizer(BONE_CFG, BONE_CKPT, device=str(device))
    bone_model.eval()
    print("✅ Bone model loaded")

    # YOLO pose
    yolo = YOLO("yolo11n-pose.pt")
    print("✅ YOLO11n-pose loaded")

    return rgb_model, joint_model, bone_model, yolo


# ── RGB transform ─────────────────────────────────────────────
rgb_tf = T.Compose([
    T.Resize((112, 112)),
    T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                std =[0.22803,  0.22145,  0.216989])
])


# ── Helper functions ──────────────────────────────────────────
def choose_top2_persons(boxes_xyxy, keypoints_conf, max_persons=2):
    """Select top-2 most salient persons: salience = bbox_area x mean_kpt_conf."""
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


def get_rgb_probs(video_path, rgb_model, num_frames=16):
    """Get RGB stream softmax probabilities using random whole-video sampling."""
    vr  = VideoReader(video_path, ctx=cpu(0))
    idx = sorted(random.sample(range(len(vr)), min(num_frames, len(vr))))
    frames = vr.get_batch(idx).asnumpy()
    frames = torch.from_numpy(frames).float() / 255.0
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
    clip   = torch.stack([rgb_tf(frames[t]) for t in range(frames.shape[0])])
    clip   = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # (1, C, T, H, W)
    with torch.no_grad():
        logits = rgb_model(clip)
    return F.softmax(logits, dim=1).cpu().numpy()[0]


def extract_skeleton(video_path, yolo_model, conf=0.25, imgsz=640, max_frames=150):
    """
    Extract YOLO11n-pose keypoints from video.
    Returns skeleton dict compatible with MMAction2 inference_recognizer.
    """
    cap = cv2.VideoCapture(video_path)
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_indices = np.linspace(0, total - 1, max_frames, dtype=int)
    saved = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        frame_kpts = np.zeros((2, 17, 3), dtype=np.float32)

        if ok:
            results = yolo_model.predict(source=frame, conf=conf,
                                          imgsz=imgsz, verbose=False)
            if len(results) > 0:
                r = results[0]
                if (hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0
                        and hasattr(r, "keypoints") and r.keypoints is not None):
                    boxes_xyxy     = r.boxes.xyxy.cpu().numpy()
                    keypoints_xy   = r.keypoints.xy.cpu().numpy()
                    keypoints_conf = r.keypoints.conf.cpu().numpy() if getattr(r.keypoints, "conf", None) is not None else None
                    top_idxs       = choose_top2_persons(boxes_xyxy, keypoints_conf)
                    for slot, idx2 in enumerate(top_idxs):
                        frame_kpts[slot, :, :2] = keypoints_xy[idx2][:17]
                        frame_kpts[slot, :, 2]  = keypoints_conf[idx2][:17] if keypoints_conf is not None else 1.0

        saved.append(frame_kpts)
    cap.release()

    arr            = np.stack(saved, axis=0).astype(np.float32)  # (T, 2, 17, 3)
    keypoint       = arr[:, :, :, :2].transpose(1, 0, 2, 3)      # (2, T, 17, 2)
    keypoint_score = arr[:, :, :, 2].transpose(1, 0, 2)          # (2, T, 17)

    return {
        "keypoint":       keypoint,
        "keypoint_score": keypoint_score,
        "img_shape":      (h, w),
        "total_frames":   max_frames,
        "frame_dir":      video_path,
        "label":          -1,
    }


def get_agcn_probs(model, skeleton_data):
    """Get 2s-AGCN softmax probabilities using MMAction2 inference_recognizer."""
    from mmaction.apis import inference_recognizer
    result = inference_recognizer(model, skeleton_data)
    return F.softmax(torch.tensor(result.pred_score), dim=0).cpu().numpy()


# ── Main predict function ─────────────────────────────────────
def predict(video_path, rgb_model, joint_model, bone_model, yolo_model):
    """
    Run full three-stream late fusion pipeline on a video.
    Returns formatted markdown string with prediction and scores.
    """
    if video_path is None:
        return "Please upload a video."

    try:
        # RGB stream
        rgb_probs = get_rgb_probs(video_path, rgb_model)

        # Pose extraction
        skeleton = extract_skeleton(video_path, yolo_model)

        # Skeleton streams
        joint_probs = get_agcn_probs(joint_model, copy.deepcopy(skeleton))
        bone_probs  = get_agcn_probs(bone_model,  copy.deepcopy(skeleton))

        # Late fusion
        fused = W_RGB * rgb_probs + W_JOINT * joint_probs + W_BONE * bone_probs
        pred  = int(np.argmax(fused))
        label = CLASS_NAMES[pred]
        conf  = float(fused[pred])

        return f"""
## Prediction: **{label}**
### Confidence: **{conf*100:.2f}%**

| Stream | NonFight | Fight |
|--------|----------|-------|
| RGB (w={W_RGB}) | {rgb_probs[0]:.4f} | {rgb_probs[1]:.4f} |
| Joint (w={W_JOINT}) | {joint_probs[0]:.4f} | {joint_probs[1]:.4f} |
| Bone (w={W_BONE}) | {bone_probs[0]:.4f} | {bone_probs[1]:.4f} |
| **Fusion** | **{fused[0]:.4f}** | **{fused[1]:.4f}** |
"""
    except Exception as e:
        import traceback
        return f"Error: {traceback.format_exc()}"


# ── Launch Gradio ─────────────────────────────────────────────
if __name__ == "__main__":
    # Load all models once at startup
    rgb_model, joint_model, bone_model, yolo_model = load_models()

    # Wrap predict with loaded models
    def gradio_predict(video_path):
        return predict(video_path, rgb_model, joint_model, bone_model, yolo_model)

    demo = gr.Interface(
        fn=gradio_predict,
        inputs=gr.Video(label="Upload surveillance video (.mp4 / .avi)"),
        outputs=gr.Markdown(label="Prediction"),
        title="Violence Detection in Videos",
        description=(
            "Three-stream late-fusion pipeline: R3D-18 RGB + Joint 2s-AGCN + Bone 2s-AGCN\n"
            f"Fusion weights: RGB={W_RGB} | Joint={W_JOINT} | Bone={W_BONE}"
        ),
        theme=gr.themes.Soft()
    )

    demo.launch(share=True)
