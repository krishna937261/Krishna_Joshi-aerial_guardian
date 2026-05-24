"""
Visualizer — draws bounding boxes, unique ID labels, and trajectory tails.
Colours are assigned per-track-ID using a hue wheel so each person has
a distinct colour that persists across frames.
"""

import cv2
import numpy as np
from typing import List


# Pre-compute 100 distinct BGR colours
_PALETTE = []
for i in range(100):
    hue = int(i * 180 / 100)
    hsv = np.uint8([[[hue, 220, 200]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    _PALETTE.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))


def track_color(track_id: int):
    return _PALETTE[track_id % len(_PALETTE)]


def draw_tracks(frame: np.ndarray, tracks, traj_buffer) -> np.ndarray:
    """
    frame      : BGR image
    tracks     : list of STrack objects (with .tlbr and .track_id)
    traj_buffer: TrajectoryBuffer instance
    """
    active_ids = [t.track_id for t in tracks]
    traj_buffer.prune(active_ids)

    for track in tracks:
        tid = track.track_id
        color = track_color(tid)
        x1, y1, x2, y2 = [int(v) for v in track.tlbr]

        # ── Bounding box ──
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ── ID label with background ──
        label = f"P{tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Trajectory tail ──
        pts = traj_buffer.get(tid)
        if len(pts) >= 2:
            for k in range(1, len(pts)):
                # Fade older points: linearly decrease thickness
                alpha = k / len(pts)
                thickness = max(1, int(2 * alpha))
                cv2.line(frame, pts[k - 1], pts[k], color, thickness, cv2.LINE_AA)

    return frame
