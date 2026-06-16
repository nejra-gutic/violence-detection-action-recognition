# Violence Detection in Videos Using Human Action Recognition

**Nejra Gutić & Zeynep Nur Yılmaz**  
Istanbul Technical University, Department of Computer Engineering, 2026  
Advisor: Dr. Cihan Topal

> ✅ Accepted to **TÜBİTAK 2209-A** Undergraduate Research Projects Support Programme (2025/1)

---

## Overview

A three-stream late-fusion framework for automated violence detection in surveillance videos. The system combines RGB appearance features with body-pose dynamics — avoiding optical flow entirely while surpassing optical-flow-based baselines.

**Three parallel streams:**
- **RGB Stream** — R3D-18 (3D ResNet) pretrained on Kinetics-400, fine-tuned on RWF-2000
- **Joint Stream** — 2s-AGCN on absolute keypoint (x, y) coordinates
- **Bone Stream** — 2s-AGCN on directed inter-joint vectors (limb orientation)

Pose keypoints are extracted per-frame using **YOLO11n-pose**, retaining the top-2 most salient persons per frame based on bounding-box area × mean keypoint confidence. Stream softmax scores are combined via weighted late fusion with weights optimized by exhaustive grid search.

---

## Results

### RWF-2000 Benchmark (400 val clips)

| Method | Input | Accuracy |
|--------|-------|----------|
| C3D [4] | RGB | 82.75% |
| I3D [9] | RGB + Flow | 86.50% |
| Flow Gated Network [1] | RGB + Flow | 87.25% |
| **Ours (3-stream fusion)** | **RGB + Pose** | **89.25%** |

### Ablation Study

| Model | Accuracy | F1 |
|-------|----------|----|
| RGB only (v1) | 83.00% | 0.843 |
| RGB only (v2) | 88.00% | — |
| Joint only | 80.75% | — |
| Bone only | 83.50% | — |
| Joint + Bone | 84.00% | 0.835 |
| RGB + Joint | 88.00% | 0.880 |
| RGB + Bone | 88.50% | 0.886 |
| **RGB + Joint + Bone** | **89.25%** | **0.892** |

### Zero-Shot Transfer to RLVS (1,951 clips, no retraining)

| Metric | Value |
|--------|-------|
| Accuracy | 81.19% |
| F1 Score | 0.838 |
| **Violence Recall** | **95.0%** |

---

## Pipeline

```
Input Video
    ├── RGB Stream → R3D-18 → p_RGB
    └── Pose Extraction (YOLO11n-pose, top-2 persons)
            ├── Joint Stream → 2s-AGCN → p_joint
            └── Bone Stream  → 2s-AGCN → p_bone

Late Fusion:
p_fused = 0.45 × p_RGB + 0.25 × p_joint + 0.30 × p_bone
prediction = argmax(p_fused)
```

---

## Repository Structure

```
violence-detection-action-recognition/
│
├── src/                              ← clean Python scripts
│   ├── 01_pose_extraction.py         ← YOLO11n-pose keypoint extraction
│   ├── 02_pose_preprocessing.py      ← .npy → .pkl for MMAction2
│   ├── 03_rgb_training.py            ← R3D-18 fine-tuning (v2 strategy)
│   ├── 04_skeleton_joint_stream.py   ← 2s-AGCN joint stream training
│   ├── 05_skeleton_bone_stream.py    ← 2s-AGCN bone stream training
│   └── 06_fusion.py                  ← late fusion + grid search + ablation
│
├── notebooks/                        ← original Colab notebooks
│   ├── 01_pose_extraction.ipynb
│   ├── 02_pose_preprocessing.ipynb
│   ├── 03_rgb_training_v2.ipynb      ← v2 strategy (88% accuracy)
│   ├── 03_rgb_v1.ipynb               ← v1 baseline (83% accuracy)
│   ├── 04_2s-agcn_jointStream.ipynb
│   ├── 04_2s-agcn_boneStream.ipynb
│   ├── 06_fusion_RGB + joint + bone stream.ipynb  ← main result (89.25%)
│   ├── 06_fusion_RGB+jointStream.ipynb            ← ablation
│   ├── 06_fusion_RGB+boneStream.ipynb             ← ablation
│   ├── 06_fusion_joint + bone stream.ipynb        ← ablation
│   └── evaluate_on_other_dataset.ipynb            ← RLVS zero-shot eval
│
├── demo/                             ← Gradio web interface
│   ├── gradio_demo.py
│   └── violence_detection_demo.ipynb ← Colab-ready demo notebook
│
├── results/                          ← confusion matrices, training curves
├── docs/                             ← paper PDF
├── requirements.txt
└── README.md
```

---

## How to Run

### Prerequisites

```bash
pip install torch torchvision ultralytics decord gradio mmengine
pip install mmcv  # see https://mmcv.readthedocs.io for version matching
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2 && pip install -e .
```

### Step-by-step

**Step 1 — Pose Extraction**
```bash
# Update DATA_ROOT and OUT_ROOT in the script first
python src/01_pose_extraction.py
# Output: data/processed/pose/npy/{split}/{class}/{video}.npy
```

**Step 2 — Pose Preprocessing**
```bash
python src/02_pose_preprocessing.py
# Output: data/processed/pose/pkl/rwf_pose.pkl
```

**Step 3 — RGB Training**
```bash
python src/03_rgb_training.py
# Output: checkpoints/rgb_v2/best_model.pth
#         checkpoints/rgb_v2/rgb_v2_val_probs.npy
```

**Step 4 & 5 — Skeleton Stream Training**
```bash
# Generate MMAction2 config files
python src/04_skeleton_joint_stream.py
python src/05_skeleton_bone_stream.py

# Train using MMAction2
cd /path/to/mmaction2
python tools/train.py configs/skeleton/2s-agcn/2s-agcn_rwf2000_joint.py
python tools/train.py configs/skeleton/2s-agcn/2s-agcn_rwf2000_bone.py
```

**Step 6 — Fusion & Evaluation**
```bash
python src/06_fusion.py
# Prints full ablation table and saves results
```

**Demo**
```bash
python demo/gradio_demo.py
# Opens Gradio interface at http://localhost:7860
```

---

## Datasets

- **RWF-2000**: [GitHub](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) — 2,000 surveillance clips, Fight/NonFight, used for training and validation
- **RLVS**: [Kaggle](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) — 1,951 clips, used for zero-shot cross-dataset evaluation only

---

## Key Design Decisions

**Why late fusion?**
Each stream has a different input format (video tensor vs skeleton graph) making feature-level fusion architecturally complex. Late fusion is modular — any stream can be upgraded independently without retraining the others.

**Why top-2 persons?**
Violence involves interaction between people. Keeping only the 2 most salient persons (scored by bbox area × keypoint confidence) reduces noise from bystanders while focusing on the primary interaction.

**Why bone stream outperforms joint stream?**
Bone vectors (directed differences between connected joints) are translation-invariant — they capture limb orientation regardless of where the person is in the frame. This makes them more discriminative for violent actions like punches and kicks.

**Why random whole-video frame sampling?**
Violent moments in RWF-2000 are sparse — often occupying only 1-2 seconds of a 5-second clip. Random sampling across the full video ensures these key moments are captured during training, acting as temporal data augmentation (83% → 88%).

---

## References

[1] M. Cheng, K. Cai, and M. Li, "RWF-2000: An Open Large Scale Video Database for Violence Detection," ICPR, 2021.

[2] L. Shi, Y. Zhang, J. Cheng, and H. Lu, "Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition," CVPR, 2019.

[3] D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, and M. Paluri, "A Closer Look at Spatiotemporal Convolutions for Action Recognition," CVPR, 2018.

[4] Ultralytics, "YOLO11: Real-Time Object Detection and Pose Estimation," github.com/ultralytics/ultralytics, 2024.

---

## Citation

```
Gutić, N. & Yılmaz, Z.N. (2026). Violence Detection in Videos Using Human Action Recognition.
Istanbul Technical University, Department of Computer Engineering.
TÜBİTAK 2209-A Project No: 1919B012558699
```
