# ============================================================
# Violence Detection — Step 5: Skeleton Bone Stream Training
# Nejra Gutic & Zeynep Nur Yilmaz, ITU 2026
#
# What this script does:
#   - Fine-tunes a 2s-AGCN model on RWF-2000 pose annotations
#   - Uses the BONE stream: directed difference vectors between
#     connected joints (e.g. b_LeftShoulder = j_LeftElbow - j_LeftShoulder)
#   - Captures limb orientation and speed independently of body position
#   - Model initialized from NTU RGB+D pretrained weights
#
# Why bone stream outperforms joint stream (83.50% vs 80.75%):
#   1. Translation invariance: bone vectors don't change when the
#      person moves across the frame — only limb orientation matters
#   2. Direct limb encoding: violent actions (punches, kicks) cause
#      abrupt changes in limb orientation — exactly what bone captures
#   3. Noise reduction: small keypoint errors cancel out when
#      computing the difference between connected joints
#
# Only difference from joint stream: feats=['b'] instead of feats=['j']
# This tells GenSkeFeat to compute bone vectors instead of using
# raw joint coordinates.
#
# Training details: identical to joint stream
#   - Best val checkpoint: 83.50% at epoch 8
#
# Usage (from mmaction2 directory):
#   python tools/train.py configs/skeleton/2s-agcn/2s-agcn_rwf2000_bone.py
# ============================================================

import os

# ── Paths — update these to match your local setup ───────────
BASE        = "/path/to/violence-detection"
PKL_FILE    = f"{BASE}/data/processed/pose/pkl/rwf_pose.pkl"
CKPT_DIR    = f"{BASE}/checkpoints/pose/2s-agcn-bone"
MMACTION2   = "/path/to/mmaction2"
CONFIG_PATH = f"{MMACTION2}/configs/skeleton/2s-agcn/2s-agcn_rwf2000_bone.py"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)

# ── MMAction2 Config ──────────────────────────────────────────
config = f"""
_base_ = '{MMACTION2}/configs/_base_/default_runtime.py'

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='AAGCN',
        graph_cfg=dict(layout='coco', mode='spatial'),
        gcn_attention=False),
    cls_head=dict(type='GCNHead', num_classes=2, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = '{PKL_FILE}'

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),   # bone stream
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]

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

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=80,
                 val_begin=1, val_interval=1)
val_cfg  = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [dict(type='CosineAnnealingLR', eta_min=0,
    T_max=80, by_epoch=True, convert_to_iter_based=True)]

optim_wrapper = dict(optimizer=dict(
    type='SGD', lr=0.1, momentum=0.9,
    weight_decay=0.0005, nesterov=True))

default_hooks = dict(
    checkpoint=dict(interval=1, save_best='acc/top1'),
    logger=dict(interval=50))

auto_scale_lr = dict(enable=False, base_batch_size=128)
work_dir = './work_dirs/2s-agcn_rwf2000_bone'
"""

with open(CONFIG_PATH, 'w') as f:
    f.write(config)
print(f"Config saved to: {CONFIG_PATH}")

print("\nTo train, run from the mmaction2 directory:")
print(f"  python tools/train.py {CONFIG_PATH}")
print("\nAfter training, copy the best checkpoint:")
print(f"  cp work_dirs/2s-agcn_rwf2000_bone/best_acc_top1_epoch_*.pth {CKPT_DIR}/")
