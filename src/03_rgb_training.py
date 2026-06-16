# ============================================================
# Violence Detection — Step 3: RGB Stream Training (v2)
# Nejra Gutic & Zeynep Nur Yilmaz, ITU 2026
#
# What this script does:
#   - Fine-tunes R3D-18 (3D ResNet pretrained on Kinetics-400)
#     on RWF-2000 for binary Fight/NonFight classification
#   - Uses v2 training strategy (88% val accuracy):
#       1. Random whole-video frame sampling (temporal augmentation)
#       2. 3-phase gradual backbone unfreezing
#       3. Best validation checkpoint saving
#
# Model: R3D-18 (Tran et al., CVPR 2018)
#   - 3D convolutions process spatial + temporal dimensions together
#   - Pretrained on Kinetics-400 (400 human action categories)
#   - Final FC layer replaced with 2-class head (Fight/NonFight)
#
# Why gradual unfreezing?
#   Unfreezing all layers at once causes large gradients that destroy
#   pretrained Kinetics-400 features. Gradual unfreezing lets the
#   classification head stabilize first, then adapts deeper layers.
#
# Why random whole-video frame sampling?
#   Violent moments in RWF-2000 are sparse — they may only occupy
#   1-2 seconds of a 5-second clip. Consecutive window sampling
#   often misses the key moment entirely. Random sampling across
#   the full video almost always captures some violent frames.
#
# Training phases:
#   Phase 1 (epochs  1-5):  frozen backbone, only FC head trains  lr=1e-4
#   Phase 2 (epochs  6-12): last residual block unfrozen           lr=1e-4
#   Phase 3 (epochs 13-20): full network fine-tuned                lr=1e-5
#
# Results:
#   v1 (consecutive sampling, 2-phase): 83.0% val accuracy
#   v2 (random sampling, 3-phase):      88.0% val accuracy
#
# After training, run the inference section to save val probabilities
# as rgb_v2_val_probs.npy — needed for the fusion notebook.
# ============================================================

import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import pickle
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18
from decord import VideoReader, cpu
from sklearn.metrics import accuracy_score, f1_score

# ── Paths — update these to match your local setup ───────────
DATA_ROOT = "data/rwf2000_clean"          # RGB videos
SAVE_DIR  = "checkpoints/rgb_v2"          # where to save checkpoints
PKL_PATH  = "data/processed/pose/pkl/rwf_pose.pkl"  # for val ordering

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Training hyperparameters ──────────────────────────────────
NUM_FRAMES  = 16     # frames sampled per clip
IMG_SIZE    = 224    # resize frames to 224x224
BATCH_SIZE  = 8
NUM_WORKERS = 2

# Phase durations
PHASE1_EPOCHS = 5    # freeze backbone, train FC only
PHASE2_EPOCHS = 7    # unfreeze last residual block
PHASE3_EPOCHS = 8    # unfreeze full network with lower lr
TOTAL_EPOCHS  = PHASE1_EPOCHS + PHASE2_EPOCHS + PHASE3_EPOCHS  # 20

LR_PHASE1_2 = 1e-4  # learning rate for phases 1 and 2
LR_PHASE3   = 1e-5  # smaller lr for full fine-tuning phase


# ── Dataset ───────────────────────────────────────────────────
class RWF2000Dataset(Dataset):
    """
    RWF-2000 video dataset for RGB stream training.

    In train mode: randomly samples NUM_FRAMES from the entire video
    (whole-video random sampling = temporal data augmentation).
    In val mode: uniformly samples NUM_FRAMES across the video.
    """
    def __init__(self, root, split, num_frames=16, size=224, mode="train"):
        self.num_frames = num_frames
        self.mode       = mode

        # Normalization stats from Kinetics-400 (matches R3D-18 pretraining)
        normalize = T.Normalize(
            mean=[0.43216, 0.394666, 0.37645],
            std =[0.22803,  0.22145,  0.216989]
        )

        if mode == "train":
            self.tf = T.Compose([
                T.Resize((size, size)),
                T.RandomHorizontalFlip(0.5),
                normalize,
            ])
        else:
            self.tf = T.Compose([
                T.Resize((size, size)),
                normalize,
            ])

        self.class_to_idx = {"NonFight": 0, "Fight": 1}
        self.samples = []
        for cls, label in self.class_to_idx.items():
            folder = os.path.join(root, split, cls)
            vids   = sorted(glob(os.path.join(folder, "*.avi")))
            self.samples += [(v, label) for v in vids]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        vr = VideoReader(path, ctx=cpu(0))
        n  = len(vr)

        if self.mode == "train":
            # Random whole-video sampling (v2 key improvement)
            # Captures violent moments regardless of when they occur
            indices = sorted(random.sample(range(n), min(self.num_frames, n)))
            while len(indices) < self.num_frames:
                indices.append(indices[-1])  # pad if video too short
        else:
            # Uniform sampling for consistent validation
            indices = torch.linspace(0, n - 1, self.num_frames).long().tolist()

        frames = vr.get_batch(indices).asnumpy()
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        frames = torch.stack([self.tf(frames[t]) for t in range(frames.shape[0])])
        clip   = frames.permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)

        return clip, torch.tensor(label)


# ── Model builder ─────────────────────────────────────────────
def build_model(device):
    """
    Build R3D-18 with Kinetics-400 pretrained weights.
    Replace final FC layer with 2-class head for Fight/NonFight.
    """
    model = r3d_18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


# ── Training helpers ──────────────────────────────────────────
def set_phase(model, optimizer, phase):
    """
    Configure which layers are trainable for each training phase.

    Phase 1: freeze backbone, train only FC head
    Phase 2: unfreeze last residual block (layer4)
    Phase 3: unfreeze full network with reduced learning rate
    """
    if phase == 1:
        # Freeze all backbone layers
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        for pg in optimizer.param_groups:
            pg['lr'] = LR_PHASE1_2
        print("Phase 1: backbone frozen, training FC head only")

    elif phase == 2:
        # Unfreeze last residual block
        for param in model.layer4.parameters():
            param.requires_grad = True
        for pg in optimizer.param_groups:
            pg['lr'] = LR_PHASE1_2
        print("Phase 2: unfroze layer4 (last residual block)")

    elif phase == 3:
        # Unfreeze full network with smaller lr
        for param in model.parameters():
            param.requires_grad = True
        for pg in optimizer.param_groups:
            pg['lr'] = LR_PHASE3
        print("Phase 3: full network fine-tuning with lr=1e-5")


def run_epoch(model, loader, criterion, optimizer=None, device="cpu", train=True):
    """Run one training or validation epoch."""
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    for clips, labels in tqdm(loader, desc="Train" if train else "Val", leave=False):
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            outputs = model(clips)
            loss    = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * clips.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds)

    return avg_loss, acc, f1


# ── Main training loop ────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Datasets and loaders
    train_ds = RWF2000Dataset(DATA_ROOT, "train", NUM_FRAMES, IMG_SIZE, "train")
    val_ds   = RWF2000Dataset(DATA_ROOT, "val",   NUM_FRAMES, IMG_SIZE, "val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True)

    print(f"Train: {len(train_ds)} videos | Val: {len(val_ds)} videos")

    # Model, loss, optimizer
    model     = build_model(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_PHASE1_2)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, TOTAL_EPOCHS + 1):

        # Set training phase
        if epoch == 1:
            set_phase(model, optimizer, phase=1)
        elif epoch == PHASE1_EPOCHS + 1:
            set_phase(model, optimizer, phase=2)
        elif epoch == PHASE1_EPOCHS + PHASE2_EPOCHS + 1:
            set_phase(model, optimizer, phase=3)

        # Train and validate
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, None, device, train=False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{TOTAL_EPOCHS} | "
              f"Train loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"Val loss={val_loss:.4f} acc={val_acc:.3f} f1={val_f1:.3f}")

        # Save every epoch checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
        }, os.path.join(SAVE_DIR, f"r3d_epoch_{epoch}.pth"))

        # Save best val checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"  ★ New best val acc: {best_val_acc:.4f} — saved best_model.pth")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    return model


# ── Inference for fusion ──────────────────────────────────────
def run_inference_for_fusion(device):
    """
    Run the best model on the validation set and save softmax probabilities.
    The val set is ordered according to the pose .pkl file to ensure
    alignment with skeleton stream predictions during fusion.

    Output: SAVE_DIR/rgb_v2_val_probs.npy  — shape (400, 2)
            SAVE_DIR/rgb_v2_val_labels.npy — shape (400,)
    """
    # Load best model
    model = build_model(device)
    model.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "best_model.pth"),
                   map_location=device, weights_only=False))
    model.eval()
    print("Best model loaded for inference.")

    # Load val order from pose pkl to align with skeleton predictions
    with open(PKL_PATH, "rb") as f:
        pose_data = pickle.load(f)
    val_order = pose_data["split"]["val"]
    print(f"Val order: {len(val_order)} videos")

    # Build ordered val dataset
    tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                    std =[0.22803,  0.22145,  0.216989]),
    ])

    samples = []
    for frame_dir in val_order:
        split, cls, name = frame_dir.split("/")
        path  = os.path.join(DATA_ROOT, split, cls, f"{name}.avi")
        label = 1 if cls == "Fight" else 0
        samples.append((path, label))

    # Run inference
    rgb_probs_list, rgb_labels_list = [], []

    with torch.no_grad():
        for i in range(0, len(samples), BATCH_SIZE):
            batch_samples = samples[i:i + BATCH_SIZE]
            batch_clips, batch_labels = [], []

            for path, label in batch_samples:
                vr     = VideoReader(path, ctx=cpu(0))
                inds   = torch.linspace(0, len(vr)-1, NUM_FRAMES).long().tolist()
                frames = vr.get_batch(inds).asnumpy()
                frames = torch.from_numpy(frames).float() / 255.0
                frames = frames.permute(0, 3, 1, 2)
                frames = torch.stack([tf(frames[t]) for t in range(frames.shape[0])])
                batch_clips.append(frames.permute(1, 0, 2, 3))
                batch_labels.append(label)

            clips = torch.stack(batch_clips).to(device)
            probs = torch.softmax(model(clips), dim=1).cpu().numpy()
            rgb_probs_list.extend(probs)
            rgb_labels_list.extend(batch_labels)

    rgb_probs_arr  = np.array(rgb_probs_list)
    rgb_labels_arr = np.array(rgb_labels_list)

    acc = accuracy_score(rgb_labels_arr, np.argmax(rgb_probs_arr, axis=1))
    print(f"RGB v2 val accuracy: {acc:.4f}")

    np.save(os.path.join(SAVE_DIR, "rgb_v2_val_probs.npy"),  rgb_probs_arr)
    np.save(os.path.join(SAVE_DIR, "rgb_v2_val_labels.npy"), rgb_labels_arr)
    print("Saved — ready for fusion script.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Train
    train()

    # Step 2: Save val probabilities for fusion
    run_inference_for_fusion(device)
