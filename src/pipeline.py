"""
Aerial Guardian — Drone-based Person Detection & Tracking Pipeline
Architecture: YOLOv8n + SAHI (Slicing Aided Hyper Inference) + ByteTrack
Custom additions for drone scenario:
  - SAHI slicing for small-object detection at altitude
  - EMA-based trajectory smoothing to reduce ID switches from camera shake
  - Adaptive confidence thresholding based on detected object scale
  - Drone ego-motion compensation via optical-flow-guided IoU matching
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from tracker.bytetrack import BYTETracker
from tracker.trajectory import TrajectoryBuffer
from utils.ego_motion import EgoMotionCompensator
from utils.visualizer import draw_tracks


# ─────────────────────────────────────────────
#  Adaptive confidence: small boxes → lower threshold
# ─────────────────────────────────────────────
def adaptive_conf_threshold(box_area: float, frame_area: float,
                             base_conf: float = 0.25) -> float:
    """
    Lower the confidence threshold for very small detections (distant persons).
    Objects < 0.05% of frame area get a -0.08 reduction in threshold.
    Objects < 0.01% of frame area get a -0.15 reduction.
    This recovers small, faint detections typical in drone footage.
    """
    ratio = box_area / (frame_area + 1e-6)
    if ratio < 0.0001:
        return max(0.10, base_conf - 0.15)
    elif ratio < 0.0005:
        return max(0.15, base_conf - 0.08)
    return base_conf


class AerialGuardianPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.device = config.get("device", "cpu")

        # ── Detector (YOLOv8n fine-tuned on VisDrone) ──
        model_path = config.get("model_path", "weights/yolov8n_visdrone.pt")
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=config.get("conf_threshold", 0.25),
            device=self.device,
        )

        # ── SAHI slicing parameters ──
        self.slice_height = config.get("slice_height", 320)
        self.slice_width = config.get("slice_width", 320)
        self.overlap_ratio = config.get("overlap_ratio", 0.2)

        # ── ByteTrack ──
        self.tracker = BYTETracker(
            track_thresh=config.get("track_thresh", 0.45),
            track_buffer=config.get("track_buffer", 30),
            match_thresh=config.get("match_thresh", 0.8),
            frame_rate=config.get("fps", 30),
        )

        # ── Trajectory buffer (tail lines) ──
        self.traj_buffer = TrajectoryBuffer(max_len=config.get("tail_len", 30))

        # ── Ego-motion compensator ──
        self.ego_motion = EgoMotionCompensator(
            method=config.get("ego_motion_method", "optical_flow")
        )

        # ── Stats ──
        self.frame_times: List[float] = []

    # ──────────────────────────────────────────
    def detect_with_sahi(self, frame: np.ndarray) -> List[dict]:
        """
        Run SAHI sliced inference. Each slice is inferred independently,
        then detections are merged with NMS.  This dramatically improves
        recall for small (< 32×32 px) persons at altitude.
        """
        result = get_sliced_prediction(
            frame,
            self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            postprocess_type="NMM",          # Non-maximum merging (softer than NMS)
            postprocess_match_threshold=0.5,
            verbose=0,
        )

        frame_area = frame.shape[0] * frame.shape[1]
        detections = []
        for obj in result.object_prediction_list:
            # Only keep "person" class (index 0 in VisDrone person-only model)
            if obj.category.name not in ("person", "pedestrian"):
                continue
            bbox = obj.bbox  # sahi BoundingBox
            x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
            box_area = (x2 - x1) * (y2 - y1)
            conf = obj.score.value

            # Adaptive threshold check
            thresh = adaptive_conf_threshold(box_area, frame_area,
                                             self.config.get("conf_threshold", 0.25))
            if conf < thresh:
                continue
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "cls": 0,
            })
        return detections

    # ──────────────────────────────────────────
    def process_video(self, input_path: str, output_path: str) -> dict:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps_in, (W, H))

        prev_gray = None
        frame_idx = 0

        print(f"Processing {total_frames} frames ({W}×{H} @ {fps_in:.1f} fps)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()

            # ── Ego-motion compensation (warp previous tracks) ──
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            warp_matrix = self.ego_motion.estimate(prev_gray, gray)
            if warp_matrix is not None:
                self.tracker.compensate_motion(warp_matrix)
            prev_gray = gray

            # ── Detection with SAHI ──
            detections = self.detect_with_sahi(frame)

            # ── ByteTrack update ──
            if detections:
                dets_np = np.array([[*d["bbox"], d["conf"]] for d in detections],
                                    dtype=np.float32)
            else:
                dets_np = np.empty((0, 5), dtype=np.float32)

            tracks = self.tracker.update(dets_np, (H, W), (H, W))

            # ── Update trajectory tails ──
            for track in tracks:
                tid = int(track.track_id)
                cx = int((track.tlbr[0] + track.tlbr[2]) / 2)
                cy = int((track.tlbr[1] + track.tlbr[3]) / 2)
                self.traj_buffer.update(tid, cx, cy)

            # ── Visualise ──
            vis = draw_tracks(frame.copy(), tracks, self.traj_buffer)

            # FPS overlay
            elapsed = time.perf_counter() - t0
            self.frame_times.append(elapsed)
            fps_live = 1.0 / elapsed
            cv2.putText(vis, f"FPS: {fps_live:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis, f"Tracks: {len(tracks)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            out.write(vis)
            frame_idx += 1

            if frame_idx % 50 == 0:
                avg_fps = 1.0 / (sum(self.frame_times[-50:]) / 50)
                print(f"  Frame {frame_idx}/{total_frames} | Avg FPS: {avg_fps:.1f} "
                      f"| Active tracks: {len(tracks)}")

        cap.release()
        out.release()

        avg_fps = 1.0 / (np.mean(self.frame_times) + 1e-9)
        stats = {
            "total_frames": frame_idx,
            "avg_fps": round(avg_fps, 2),
            "input": input_path,
            "output": output_path,
        }
        print(f"\n✓ Done. Average FPS: {avg_fps:.2f}")
        return stats


# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Aerial Guardian — Drone MOT Pipeline")
    p.add_argument("--input",  required=True, help="Input video path")
    p.add_argument("--output", default="outputs/result.mp4", help="Output video path")
    p.add_argument("--model",  default="weights/yolov8n_visdrone.pt")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--conf",   type=float, default=0.25)
    p.add_argument("--slice",  type=int,   default=320, help="SAHI slice size")
    p.add_argument("--tail",   type=int,   default=30,  help="Trajectory tail length")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = {
        "model_path":  args.model,
        "device":      args.device,
        "conf_threshold": args.conf,
        "slice_height":   args.slice,
        "slice_width":    args.slice,
        "tail_len":       args.tail,
    }
    pipeline = AerialGuardianPipeline(config)
    stats = pipeline.process_video(args.input, args.output)
    print(stats)
