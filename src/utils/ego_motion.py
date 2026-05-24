"""
EgoMotionCompensator — estimates and compensates for drone camera motion.

When a drone moves, every pixel in the frame translates/rotates.  ByteTrack's
IoU-based association breaks when predicted track positions no longer overlap
the detection boxes because the camera has panned.

Fix: estimate the frame-to-frame homography (H) using sparse optical flow on
background feature points, then warp all Kalman-predicted track centroids
by H^-1 (back to the "camera reference frame") before running assignment.

Two methods:
  - optical_flow  (default): fast Lucas-Kanade on Shi-Tomasi corners
  - ecc           : Enhanced Correlation Coefficient — more robust but ~3x slower
"""

import cv2
import numpy as np
from typing import Optional


class EgoMotionCompensator:
    def __init__(self, method: str = "optical_flow",
                 max_corners: int = 200,
                 quality_level: float = 0.01,
                 min_distance: int = 20):
        assert method in ("optical_flow", "ecc"), f"Unknown method: {method}"
        self.method = method
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance

        # ECC termination criteria
        self._ecc_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1e-4)

    def estimate(self, prev_gray: Optional[np.ndarray],
                 curr_gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns a 3×3 homography (or None if estimation fails / first frame).
        The returned matrix maps points in prev_frame → curr_frame.
        """
        if prev_gray is None:
            return None

        if self.method == "optical_flow":
            return self._optical_flow(prev_gray, curr_gray)
        else:
            return self._ecc(prev_gray, curr_gray)

    def _optical_flow(self, prev, curr) -> Optional[np.ndarray]:
        # Detect good features in previous frame (background only — no masks here,
        # but in practice person boxes are tiny so they barely contribute)
        pts0 = cv2.goodFeaturesToTrack(
            prev, maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance)

        if pts0 is None or len(pts0) < 10:
            return None

        pts1, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, pts0, None)

        good0 = pts0[status.ravel() == 1]
        good1 = pts1[status.ravel() == 1]

        if len(good0) < 8:
            return None

        H, mask = cv2.findHomography(good0, good1, cv2.RANSAC, 3.0)
        return H  # 3×3 or None

    def _ecc(self, prev, curr) -> Optional[np.ndarray]:
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(prev, curr, warp,
                                            cv2.MOTION_EUCLIDEAN,
                                            self._ecc_criteria)
        except cv2.error:
            return None
        # Embed 2×3 affine into 3×3 homography
        H = np.eye(3, dtype=np.float64)
        H[:2] = warp.astype(np.float64)
        return H
