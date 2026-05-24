"""
evaluate.py — Evaluate the pipeline on VisDrone MOT validation set.

Computes standard MOT metrics using the `motmetrics` library:
  - MOTA  (Multi-Object Tracking Accuracy)
  - MOTP  (Multi-Object Tracking Precision)
  - IDF1  (ID F1 Score)
  - MT/ML (Mostly Tracked / Mostly Lost)
  - ID Sw (ID Switches — the key metric for drone footage)

Usage:
    python evaluate.py \
        --visdrone_root /data/VisDrone/VisDrone2019-MOT-val \
        --model weights/yolov8n_visdrone.pt \
        --device cpu
"""

import argparse
import os
from pathlib import Path
import numpy as np
import cv2
import motmetrics as mm
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import AerialGuardianPipeline

VISDRONE_PERSON_CLASSES = {1, 2}


def load_gt(ann_file: str):
    """Returns dict: frame_idx → list of (track_id, x1, y1, x2, y2)"""
    gt = {}
    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            frame_idx = int(parts[0])
            track_id  = int(parts[1])
            cls       = int(parts[7])
            occ       = int(parts[9]) if len(parts) > 9 else 0
            if cls not in VISDRONE_PERSON_CLASSES:
                continue
            if occ == 2:   # heavily occluded → skip
                continue
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            gt.setdefault(frame_idx, []).append((track_id, x, y, x + w, y + h))
    return gt


def evaluate(args):
    config = {
        "model_path":     args.model,
        "device":         args.device,
        "conf_threshold": 0.25,
        "slice_height":   320,
        "slice_width":    320,
        "tail_len":       0,   # no visualisation during eval
    }
    pipeline = AerialGuardianPipeline(config)

    acc = mm.MOTAccumulator(auto_id=True)
    src = Path(args.visdrone_root)
    seq_dirs = sorted((src / "sequences").iterdir())
    ann_dir  = src / "annotations"

    for seq in tqdm(seq_dirs, desc="Evaluating sequences"):
        ann_file = ann_dir / f"{seq.name}.txt"
        if not ann_file.exists():
            continue

        gt = load_gt(str(ann_file))
        img_files = sorted((seq / "img1").glob("*.jpg")) if (seq / "img1").exists() \
                    else sorted(seq.glob("*.jpg"))

        prev_gray = None
        pipeline.tracker = pipeline.tracker.__class__(
            track_thresh=0.45, track_buffer=30, match_thresh=0.8, frame_rate=25)

        for img_path in img_files:
            frame_idx = int(img_path.stem)
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            H, W = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Ego-motion compensation
            warp = pipeline.ego_motion.estimate(prev_gray, gray)
            if warp is not None:
                pipeline.tracker.compensate_motion(warp)
            prev_gray = gray

            # Detect
            dets = pipeline.detect_with_sahi(frame)
            if dets:
                dets_np = np.array([[*d["bbox"], d["conf"]] for d in dets], dtype=np.float32)
            else:
                dets_np = np.empty((0, 5), dtype=np.float32)

            tracks = pipeline.tracker.update(dets_np, (H, W), (H, W))

            # Ground truth for this frame
            gt_frame = gt.get(frame_idx, [])
            gt_ids   = [g[0] for g in gt_frame]
            gt_boxes = np.array([[g[1], g[2], g[3], g[4]] for g in gt_frame]) if gt_frame else np.empty((0, 4))

            # Predicted
            pred_ids  = [t.track_id for t in tracks]
            pred_boxes = np.array([t.tlbr for t in tracks]) if tracks else np.empty((0, 4))

            # Compute pairwise IoU distance
            if len(gt_boxes) and len(pred_boxes):
                dist = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
            else:
                dist = np.empty((len(gt_ids), len(pred_ids)))

            acc.update(gt_ids, pred_ids, dist)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["mota", "motp", "idf1", "num_switches",
                                        "mostly_tracked", "mostly_lost", "num_false_positives",
                                        "num_misses"], name="AerialGuardian")
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--visdrone_root", required=True)
    p.add_argument("--model",  default="weights/yolov8n_visdrone.pt")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
