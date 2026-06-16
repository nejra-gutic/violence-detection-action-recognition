# ============================================================
# Violence Detection — Step 4: Skeleton Joint Stream Training
# Nejra Gutic & Zeynep Nur Yilmaz, ITU 2026
#
# What this script does:
#   - Fine-tunes a 2s-AGCN model on RWF-2000 pose annotations
#   - Uses the JOINT stream: absolute (x, y) keypoint coordinates
#     as node features on the skeleton graph
#   - Model is initialized from NTU RGB+D pretrained weights
#     (transfer learning → fine-tuning for violence detection)
#   - Saves the best validation checkpoint
#
# What is 2s-AGCN?
#   Two-Stream Adaptive Graph Convolutional Network (Shi et al., CVPR 2019)
#   Models the human skeleton as a graph:
#     - Nodes = 17 COCO keypoints (joints)
#     - Edges = bones connecting them
#   "Adaptive" means it learns additional connections between
#   non-physically-connected joints that are relevant for the action.
#
# Joint stream vs Bone stream:
#   Joint stream: uses absolute (x,y) positions → "where is each body part?"
#   Bone stream:  uses limb direction vectors  → "how is each limb oriented?"
#
# Training details:
#   - Optimizer: SGD, lr=0.1, momentum=0.9, weight_decay=5e-4
#   - Scheduler: CosineAnnealingLR over 80 epochs
#   - Batch size: 16
#   - Input: clip_len=100 frames sampled from each video
#   - Best val checkpoint saved (achieved 80.75% at epoch 8)
#
# Requirements:
#   - MMAction2 installed (https://github.com/open-mmlab/mmaction2)
#   - mmcv, mmengine installed
#   - rwf_pose.pkl from step 02_pose_preprocessing.py
#
# Usage (from mmaction2 directory):
#   python tools/train.py configs/skeleton/2s-agcn/2s-agcn_rwf2000_joint.py
# ============================================================

# NOTE: This script generates the MMAction2 config file and then
# launches training using MMAction2's train tool.
# Run this script to create the config, then run the training command above.

import os

# ── Paths — update these to match your local setup ───────────
BASE        = "/path/to/violence-detection"   # your project root
PKL_FILE    = f"{BASE}/data/processed/pose/pkl/rwf_pose.pkl"
CKPT_DIR    = f"{BASE}/checkpoints/pose/2s-agcn-joint"
MMACTION2   = "/path/to/mmaction2"            # mmaction2 repo root
CONFIG_PATH = f"{MMACTION2}/configs/skeleton/2s-agcn/2s-agcn_rwf2000_joint.py"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)

# ── MMAction2 Config ──────────────────────────────────────────
# This config defines the model architecture, data pipeline, and
# training hyperparameters for the joint stream.
config = f"""
_base_ = '{MMACTION2}/configs/_base_/default_runtime.py'

# ── Model ─────────────────────────────────────────────────────
# AAGCN with gcn_attention=False degrades to standard AGCN
# GCNHead outputs 2 classes: NonFight (0) and Fight (1)
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='AAGCN',
        graph_cfg=dict(layout='coco', mode='spatial'),
        gcn_attention=False),
    cls_head=dict(type='GCNHead', num_classes=2, in_channels=256))

# ── Data Pipeline ─────────────────────────────────────────────
# GenSkeFeat with feats=['j'] extracts joint (x,y) coordinates
# UniformSampleFrames samples clip_len=100 frames from each video
# FormatGCNInput reshapes to (batch, channels, frames, joints, persons)
dataset_type = 'PoseDataset'
ann_file = '{PKL_FILE}'

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),   # joint stream
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]

# ── Dataloaders ───────────────────────────────────────────────
train_dataloader = dict(
    batch_size=16, num_workers=2, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='RepeatDataset', times=5, dataset=dict(
        type=dataset_type, ann_file=ann_file,
        pipeline=train_pipeline, split='train')))

val_dataloader = dict(
    batch_size=16, num_workers=2, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, ann_file=ann_file,
        pipeline=val_pipeline, split='val', test_mode=True))

test_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, ann_file=ann_file,
        pipeline=test_pipeline, split='val', test_mode=True))

val_evaluator  = [dict(type='AccMetric')]
test_evaluator = val_evaluator

# ── Training Schedule ─────────────────────────────────────────
# CosineAnnealingLR: lr starts at 0.1 and smoothly decreases to ~0
# over 80 epochs — smoother than step decay, helps convergence
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80,
                 val_begin=1, val_interval=1)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [dict(type='CosineAnnealingLR', eta_min=0,
    T_max=80, by_epoch=True, convert_to_iter_based=True)]

# SGD with momentum — standard for GCN skeleton models
optim_wrapper = dict(optimizer=dict(
    type='SGD', lr=0.1, momentum=0.9,
    weight_decay=0.0005, nesterov=True))

default_hooks = dict(
    checkpoint=dict(interval=1, save_best='acc/top1'),
    logger=dict(interval=50))

auto_scale_lr = dict(enable=False, base_batch_size=128)

# ── Work directory ────────────────────────────────────────────
work_dir = './work_dirs/2s-agcn_rwf2000_joint'
"""

# Write config file
with open(CONFIG_PATH, 'w') as f:
    f.write(config)
print(f"Config saved to: {CONFIG_PATH}")

print("\nTo train, run from the mmaction2 directory:")
print(f"  python tools/train.py {CONFIG_PATH}")
print("\nAfter training, copy the best checkpoint:")
print(f"  cp work_dirs/2s-agcn_rwf2000_joint/best_acc_top1_epoch_*.pth {CKPT_DIR}/")
