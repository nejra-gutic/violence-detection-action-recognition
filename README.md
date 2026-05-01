# Violence Detection in Surveillance Videos using Action Recognition

## Overview

This project focuses on automatic violence detection in surveillance videos using deep learning-based human action recognition methods. The system classifies videos as either **Fight** or **NonFight** using both RGB-based and skeleton-based approaches.

The project was developed as a graduation project at Istanbul Technical University and explores multiple video understanding techniques, including:

* RGB-based video classification using 3D CNNs
* Skeleton-based action recognition using 2s-AGCN
* Pose extraction and preprocessing pipelines
* Late fusion of RGB and skeleton streams
* Performance comparison between different modalities

The main dataset used throughout the project is the **RWF-2000** violence detection dataset.

---

# Project Structure

```text
violence-detection-action-recognition/
│
├── notebooks/
│   ├── 01_pose_extraction.ipynb
│   ├── 02_pose_preprocessing.ipynb
│   ├── 03_rgb_training.ipynb
│   ├── 04_2s-agcn_jointStream.ipynb
│   ├── 04_2s-agcn_boneStream.ipynb
│   ├── 06_fusion_joint + bone stream.ipynb
│   ├── 06_fusion_RGB+jointStream.ipynb
│   └── 06_fusion_RGB + joint + bone stream.ipynb
│
├── docs/
├── results/
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Dataset

## RWF-2000

RWF-2000 is a large-scale real-world violence detection dataset containing surveillance videos categorized into:

* Fight
* NonFight

Dataset statistics:

| Split      | Fight | NonFight |
| ---------- | ----: | -------: |
| Train      |   800 |      800 |
| Validation |   200 |      200 |

The dataset consists of real surveillance footage with challenging conditions such as:

* crowded scenes
* motion blur
* varying camera viewpoints
* occlusions
* low-quality recordings

---

# Methodology

## 1. Pose Extraction

Human pose keypoints are extracted from videos using pose estimation models.

The extracted skeleton information is later used for graph-based action recognition models.

Main steps:

* Video loading
* Human pose estimation
* Keypoint extraction
* Saving pose annotations

Notebook:

```text
01_pose_extraction.ipynb
```

---

## 2. Pose Preprocessing

The extracted keypoints are converted into the format required by 2s-AGCN.

Preprocessing includes:

* sequence formatting
* frame normalization
* skeleton organization
* train/validation split preparation

Notebook:

```text
02_pose_preprocessing.ipynb
```

---

## 3. RGB-Based Violence Detection

The RGB stream uses a pretrained **R3D-18** video classification model from TorchVision.

### Model

* Architecture: R3D-18
* Pretrained on: Kinetics
* Framework: PyTorch

### Training Strategy

Multi-stage fine-tuning was used:

1. Fully connected layer warm-up
2. Layer4 fine-tuning
3. Full network fine-tuning

### Input Configuration

* 16 sampled frames per clip
* Resolution: 224×224
* Batch size: 8

### Data Augmentation

* Random horizontal flip
* Color jitter
* Random grayscale

Notebook:

```text
03_rgb_training.ipynb
```

---

# Skeleton-Based Violence Detection

Skeleton-based action recognition was implemented using **2s-AGCN**.

## 4. Joint Stream

The joint stream uses raw human joint coordinates as input.

### Model

* Architecture: 2s-AGCN
* Framework: MMAction2
* Skeleton layout: COCO 17 keypoints

Notebook:

```text
04_2s-agcn_jointStream.ipynb
```

### Result

| Model                | Accuracy |
| -------------------- | -------: |
| 2s-AGCN Joint Stream |   80.75% |

---

## 5. Bone Stream

The bone stream models relationships and motion between connected joints.

Notebook:

```text
04_2s-agcn_boneStream.ipynb
```

---

# Fusion Experiments

Several late-fusion strategies were explored by combining prediction probabilities from different streams.

Fusion was performed using weighted averaging of softmax outputs.

---

## Joint + Bone Stream Fusion

Notebook:

```text
06_fusion_joint + bone stream.ipynb
```

This experiment combines:

* joint stream predictions
* bone stream predictions

using weighted late fusion.

---

## RGB + Joint Stream Fusion

Notebook:

```text
06_fusion_RGB+jointStream.ipynb
```

This experiment combines:

* RGB stream
* joint stream

---

## RGB + Joint + Bone Stream Fusion

Notebook:

```text
06_fusion_RGB + joint + bone stream.ipynb
```

This experiment combines:

* RGB stream
* joint stream
* bone stream

using weighted softmax fusion.

### Best Fusion Result

| Fusion Configuration      | Accuracy |
| ------------------------- | -------: |
| RGB + Joint + Bone Fusion |   88.25% |

### Best Weights

```text
RGB   = 0.55
Joint = 0.00
Bone  = 0.45
```

This indicates that the bone stream contributed positively to final performance, while the joint stream provided limited improvement in late fusion.

---

# Results Summary

| Model                     | Accuracy |
| ------------------------- | -------: |
| RGB Stream (R3D-18)       |   83.00% |
| 2s-AGCN Joint Stream      |   80.75% |
| RGB + Joint + Bone Fusion |   88.25% |

---

# Technologies Used

| Category         | Tools               |
| ---------------- | ------------------- |
| Deep Learning    | PyTorch, MMAction2  |
| Video Processing | Decord, OpenCV      |
| Pose Estimation  | YOLO Pose           |
| Visualization    | Matplotlib, Seaborn |
| Metrics          | scikit-learn        |
| Environment      | Google Colab        |

---

# Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

# Future Work

Possible future improvements include:

* Optical flow integration
* Real-time inference optimization
* Transformer-based video models
* Additional violence datasets
* Early violence prediction
* Multi-person tracking

---

# Authors

* Nejra Gutić
* Zeynep Nur Yılmaz

Istanbul Technical University
Department of Computer Engineering

---

# License

This project is licensed under the MIT License.
