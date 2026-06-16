# ============================================================
# Violence Detection — Step 6: Late Fusion + Grid Search
# Nejra Gutic & Zeynep Nur Yilmaz, ITU 2026
#
# What this script does:
#   - Loads precomputed softmax probabilities from all 3 streams:
#       RGB (R3D-18):        rgb_v2_val_probs.npy
#       Joint (2s-AGCN):     computed here via MMAction2 inference
#       Bone  (2s-AGCN):     computed here via MMAction2 inference
#   - Runs exhaustive grid search over all weight combinations
#     (step=0.05, 231 total combinations) to find optimal fusion weights
#   - Evaluates all stream combinations (ablation study):
#       Joint only, Bone only, RGB only
#       Joint + Bone, RGB + Joint, RGB + Bone
#       RGB + Joint + Bone (best)
#   - Prints full ablation table and confusion matrix
#
# What is late fusion?
#   Each stream independently produces a softmax probability vector:
#     p_RGB   = [P(NonFight), P(Fight)]  from R3D-18
#     p_joint = [P(NonFight), P(Fight)]  from 2s-AGCN joint
#     p_bone  = [P(NonFight), P(Fight)]  from 2s-AGCN bone
#
#   These are combined via weighted sum:
#     p_fused = w_RGB * p_RGB + w_joint * p_joint + w_bone * p_bone
#     prediction = argmax(p_fused)
#
#   Weights are constrained to sum to 1.0.
#
# Why grid search instead of training the weights?
#   With only 400 validation samples and 3 weights to tune,
#   grid search is sufficient and avoids overfitting to val set.
#   A finer step of 0.01 produced no improvement over 0.05.
#
# Results:
#   RGB only:              88.00%
#   Joint only:            80.75%
#   Bone only:             83.50%
#   Joint + Bone:          84.00%  F1=0.835
#   RGB + Joint:           88.00%  F1=0.880
#   RGB + Bone:            88.50%  F1=0.886
#   RGB + Joint + Bone:    89.25%  F1=0.892  ← best
#   Optimal weights: RGB=0.45, Joint=0.25, Bone=0.30
#
# Prerequisites:
#   - Run 03_rgb_training.py first → produces rgb_v2_val_probs.npy
#   - Run 04_skeleton_joint_stream.py + 05_skeleton_bone_stream.py
#     to get the model checkpoints
#   - MMAction2 must be installed
# ============================================================

import os
import sys
import copy
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from itertools import product
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ── Paths — update these to match your local setup ───────────
BASE        = "."
RGB_PROBS   = f"{BASE}/checkpoints/rgb_v2/rgb_v2_val_probs.npy"
RGB_LABELS  = f"{BASE}/checkpoints/rgb_v2/rgb_v2_val_labels.npy"
PKL_FILE    = f"{BASE}/data/processed/pose/pkl/rwf_pose.pkl"
JOINT_CKPT  = f"{BASE}/checkpoints/pose/2s-agcn-joint/best_acc_top1_epoch_8.pth"
BONE_CKPT   = f"{BASE}/checkpoints/pose/2s-agcn_bone/best_acc_top1_epoch_8.pth"
JOINT_CFG   = "configs/2s-agcn_rwf2000_joint.py"
BONE_CFG    = "configs/2s-agcn_rwf2000_bone.py"
MMACTION2   = "/path/to/mmaction2"   # update this

RESULTS_DIR = f"{BASE}/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Skeleton inference ────────────────────────────────────────
def run_skeleton_inference(cfg_path, ckpt_path, pkl_file, device):
    """
    Run 2s-AGCN inference on the validation set using MMAction2.
    Uses 10-clip TTA (Test Time Augmentation) for stable predictions.

    Returns:
        probs:  np.ndarray (N, 2) softmax probabilities
        labels: np.ndarray (N,)   ground truth labels
    """
    sys.path.insert(0, MMACTION2)
    from mmaction.apis import init_recognizer, inference_recognizer
    from mmaction.utils import register_all_modules
    register_all_modules()

    model = init_recognizer(cfg_path, ckpt_path, device=str(device))
    model.eval()

    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    val_annos  = [a for a in data["annotations"] if a["frame_dir"] in data["split"]["val"]]
    probs_list  = []
    labels_list = []

    print(f"Running inference on {len(val_annos)} val videos...")
    for anno in val_annos:
        result = inference_recognizer(model, copy.deepcopy(anno))
        score  = F.softmax(torch.tensor(result.pred_score), dim=0).cpu().numpy()
        probs_list.append(score)
        labels_list.append(anno["label"])

    return np.array(probs_list), np.array(labels_list)


# ── Grid search ───────────────────────────────────────────────
def grid_search(probs_dict, labels, step=0.05):
    """
    Exhaustive grid search over fusion weights.
    Tests all (w1, w2, w3) combinations where w1+w2+w3=1.0
    with given step size.

    Args:
        probs_dict: dict of stream_name -> np.ndarray (N, 2)
        labels:     np.ndarray (N,) ground truth
        step:       weight increment (default 0.05 → 231 combinations)

    Returns:
        best_weights, best_acc, best_f1, best_preds
    """
    stream_names = list(probs_dict.keys())
    n_streams    = len(stream_names)
    weight_range = np.arange(0, 1 + step, step)

    best_acc, best_f1, best_weights, best_preds = 0, 0, None, None
    n_combos = 0

    for combo in product(weight_range, repeat=n_streams):
        if abs(sum(combo) - 1.0) > 1e-6:
            continue
        n_combos += 1

        fused = sum(w * probs_dict[name] for w, name in zip(combo, stream_names))
        preds = np.argmax(fused, axis=1)
        acc   = accuracy_score(labels, preds)
        f1    = f1_score(labels, preds)

        if acc > best_acc or (acc == best_acc and f1 > best_f1):
            best_acc     = acc
            best_f1      = f1
            best_weights = dict(zip(stream_names, combo))
            best_preds   = preds

    print(f"  Tested {n_combos} weight combinations")
    return best_weights, best_acc, best_f1, best_preds


# ── Ablation evaluation ───────────────────────────────────────
def evaluate_all_combinations(rgb_probs, joint_probs, bone_probs, labels):
    """
    Evaluate all stream combinations for the ablation study.
    Prints a full comparison table.
    """
    combos = {
        "Joint only":          {"joint": joint_probs},
        "Bone only":           {"bone":  bone_probs},
        "RGB only":            {"rgb":   rgb_probs},
        "Joint + Bone":        {"joint": joint_probs, "bone": bone_probs},
        "RGB + Joint":         {"rgb":   rgb_probs,   "joint": joint_probs},
        "RGB + Bone":          {"rgb":   rgb_probs,   "bone":  bone_probs},
        "RGB + Joint + Bone":  {"rgb":   rgb_probs,   "joint": joint_probs, "bone": bone_probs},
    }

    print("\n" + "="*65)
    print(f"{'Model / Stream':<28} {'Accuracy':>10} {'F1':>10} {'Best Weights'}")
    print("="*65)

    results = {}
    for name, probs_dict in combos.items():
        weights, acc, f1, preds = grid_search(probs_dict, labels)
        results[name] = {"acc": acc, "f1": f1, "weights": weights, "preds": preds}

        w_str = " | ".join([f"{k}={v:.2f}" for k, v in weights.items()])
        print(f"{name:<28} {acc*100:>9.2f}% {f1:>10.3f}   {w_str}")

    print("="*65)
    return results


# ── Main ──────────────────────────────────────────────────────
def main():
    # Step 1: Load RGB probabilities (precomputed in 03_rgb_training.py)
    print("Loading RGB probabilities...")
    rgb_probs  = np.load(RGB_PROBS)
    labels     = np.load(RGB_LABELS)
    print(f"  RGB probs shape: {rgb_probs.shape}")
    print(f"  RGB accuracy: {accuracy_score(labels, np.argmax(rgb_probs, axis=1)):.4f}")

    # Step 2: Run skeleton inference
    print("\nRunning joint stream inference...")
    joint_probs, _ = run_skeleton_inference(JOINT_CFG, JOINT_CKPT, PKL_FILE, device)
    print(f"  Joint accuracy: {accuracy_score(labels, np.argmax(joint_probs, axis=1)):.4f}")

    print("\nRunning bone stream inference...")
    bone_probs, _ = run_skeleton_inference(BONE_CFG, BONE_CKPT, PKL_FILE, device)
    print(f"  Bone accuracy: {accuracy_score(labels, np.argmax(bone_probs, axis=1)):.4f}")

    # Step 3: Full ablation study
    print("\nRunning ablation study (grid search for each combination)...")
    results = evaluate_all_combinations(rgb_probs, joint_probs, bone_probs, labels)

    # Step 4: Best result details
    best = results["RGB + Joint + Bone"]
    print(f"\nBest configuration: RGB + Joint + Bone")
    print(f"  Accuracy : {best['acc']*100:.2f}%")
    print(f"  F1 Score : {best['f1']:.4f}")
    print(f"  Weights  : {best['weights']}")

    # Step 5: Confusion matrix
    cm = confusion_matrix(labels, best["preds"])
    print(f"\nConfusion Matrix (RWF-2000 val):")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    # Step 6: Save results
    save_path = os.path.join(RESULTS_DIR, "fusion_results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({
            "labels":      labels,
            "rgb_probs":   rgb_probs,
            "joint_probs": joint_probs,
            "bone_probs":  bone_probs,
            "results":     results,
        }, f)
    print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()
