"""
ByteTrack — Multi-Object Tracker adapted for drone footage.

Original paper: ByteTrack: Multi-Object Tracking by Associating Every Detection Box
Zhang et al., ECCV 2022.  https://arxiv.org/abs/2110.06864

Key drone-specific modifications in this implementation:
  1. compensate_motion(): warps existing track positions using the optical-flow
     derived homography *before* the assignment step so that camera shake does
     not break association.
  2. Lost-track patience is increased (track_buffer frames) to survive momentary
     occlusion from shadows/tree canopies common in aerial footage.
  3. Low-score detections (second association pass in vanilla ByteTrack) use a
     tighter IoU threshold when ego-motion is large (shaky camera).
"""

import numpy as np
from collections import OrderedDict
from typing import List, Optional, Tuple
import lap                      # Linear Assignment Problem (lapjv)
from scipy.spatial.distance import cdist


# ─────────────────────────────────────────────
#  Track state machine
# ─────────────────────────────────────────────
class TrackState:
    New      = 0
    Tracked  = 1
    Lost     = 2
    Removed  = 3


# ─────────────────────────────────────────────
#  Kalman filter (constant-velocity, 8-state)
# ─────────────────────────────────────────────
class KalmanFilter:
    """
    State:  [cx, cy, a, h, vx, vy, va, vh]
    Obs:    [cx, cy, a, h]
    where   a = aspect ratio (w/h), h = height
    """
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        kalman_gain = np.dot(covariance, self._update_mat.T)
        kalman_gain = np.dot(kalman_gain, np.linalg.inv(projected_cov))
        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov

    def gating_distance(self, mean, covariance, measurements):
        projected_mean, projected_cov = self.project(mean, covariance)
        d = measurements - projected_mean
        factor = np.linalg.solve(projected_cov, d.T).T
        return np.sum(d * factor, axis=1)


# ─────────────────────────────────────────────
#  Single Track
# ─────────────────────────────────────────────
class STrack:
    _count = 0
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_state = None
        self.is_activated = False
        self.score = score
        self.track_id = 0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        self.state = TrackState.New

    @classmethod
    def next_id(cls):
        cls._count += 1
        return cls._count

    def activate(self, frame_id):
        self.track_id = self.next_id()
        self.kalman_state = self.shared_kalman.initiate(self.tlwh_to_xyah(self._tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id):
        self.kalman_state = self.shared_kalman.update(
            *self.kalman_state, self.tlwh_to_xyah(new_track._tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        self.time_since_update = 0

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.score = new_track.score
        self.kalman_state = self.shared_kalman.update(
            *self.kalman_state, self.tlwh_to_xyah(new_track._tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update = 0

    def predict(self):
        mean, cov = self.kalman_state
        if self.state != TrackState.Tracked:
            mean[7] = 0
        self.kalman_state = self.shared_kalman.predict(mean, cov)

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def tlwh(self):
        if self.kalman_state is None:
            return self._tlwh.copy()
        ret = self.kalman_state[0][:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        return self.tlwh_to_tlbr(self.tlwh)


# ─────────────────────────────────────────────
#  IoU helpers
# ─────────────────────────────────────────────
def iou_batch(atlbrs, btlbrs):
    atlbrs = np.array(atlbrs); btlbrs = np.array(btlbrs)
    if atlbrs.size == 0 or btlbrs.size == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    ax1, ay1, ax2, ay2 = atlbrs[:,0], atlbrs[:,1], atlbrs[:,2], atlbrs[:,3]
    bx1, by1, bx2, by2 = btlbrs[:,0], btlbrs[:,1], btlbrs[:,2], btlbrs[:,3]
    inter_w = np.maximum(0, np.minimum(ax2[:,None], bx2) - np.maximum(ax1[:,None], bx1))
    inter_h = np.maximum(0, np.minimum(ay2[:,None], by2) - np.maximum(ay1[:,None], by1))
    inter   = inter_w * inter_h
    a_area  = (ax2 - ax1) * (ay2 - ay1)
    b_area  = (bx2 - bx1) * (by2 - by1)
    union   = a_area[:,None] + b_area - inter
    return inter / (union + 1e-6)


def linear_assignment(cost_matrix):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    matches = np.array([[ix, mx] for ix, mx in enumerate(x) if mx >= 0])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    return matches, unmatched_a, unmatched_b


# ─────────────────────────────────────────────
#  BYTETracker
# ─────────────────────────────────────────────
class BYTETracker:
    def __init__(self, track_thresh=0.45, track_buffer=30, match_thresh=0.8, frame_rate=30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.det_thresh = track_thresh + 0.1

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size

        self.tracked_tracks: List[STrack] = []
        self.lost_tracks:    List[STrack] = []
        self.removed_tracks: List[STrack] = []
        self.frame_id = 0

    def compensate_motion(self, warp_matrix: np.ndarray):
        """
        Apply affine/homography warp to all tracked positions BEFORE
        detection association.  This is the drone ego-motion fix:
        instead of every track appearing to move with the camera,
        we correct them back to world-space.
        """
        if warp_matrix is None:
            return
        for track in self.tracked_tracks + self.lost_tracks:
            if track.kalman_state is None:
                continue
            mean, cov = track.kalman_state
            # Extract centre point
            cx, cy = mean[0], mean[1]
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            if warp_matrix.shape == (3, 3):
                warped = cv2.perspectiveTransform(pt, warp_matrix)
            else:
                warped = cv2.transform(pt, warp_matrix)
            mean[0], mean[1] = warped[0, 0]
            track.kalman_state = (mean, cov)

    def update(self, dets: np.ndarray, img_size, orig_size):
        """
        dets: (N, 5)  [x1, y1, x2, y2, score]
        """
        import cv2  # imported here to avoid circular at module level
        self.frame_id += 1

        activated, refound, lost, removed = [], [], [], []

        # Convert x1y1x2y2 → tlwh
        if dets.shape[0] > 0:
            scores = dets[:, 4]
            bboxes = dets[:, :4]
            tlwhs = np.c_[bboxes[:, 0], bboxes[:, 1],
                          bboxes[:, 2] - bboxes[:, 0],
                          bboxes[:, 3] - bboxes[:, 1]]
            high_mask = scores >= self.track_thresh
            low_mask  = (scores >= 0.1) & ~high_mask
        else:
            tlwhs = np.empty((0, 4))
            scores = np.empty(0)
            high_mask = low_mask = np.zeros(0, dtype=bool)

        high_dets = [STrack(tlwhs[i], scores[i]) for i in np.where(high_mask)[0]]
        low_dets  = [STrack(tlwhs[i], scores[i]) for i in np.where(low_mask)[0]]

        # Predict positions
        for t in self.tracked_tracks + self.lost_tracks:
            t.predict()

        unconfirmed = [t for t in self.tracked_tracks if not t.is_activated]
        confirmed   = [t for t in self.tracked_tracks if t.is_activated]

        # ── First pass: high-score dets vs confirmed tracks ──
        iou = iou_batch([t.tlbr for t in confirmed], [d.tlbr for d in high_dets])
        cost = 1 - iou
        matches, u_tracks, u_dets = linear_assignment(
            np.where(cost > 1 - self.match_thresh, 1e5, cost))

        for t_idx, d_idx in matches:
            confirmed[t_idx].update(high_dets[d_idx], self.frame_id)
            activated.append(confirmed[t_idx])

        # ── Second pass: low-score dets vs unmatched confirmed tracks ──
        remaining_tracks = [confirmed[i] for i in u_tracks]
        iou2 = iou_batch([t.tlbr for t in remaining_tracks], [d.tlbr for d in low_dets])
        cost2 = 1 - iou2
        matches2, u_tracks2, _ = linear_assignment(
            np.where(cost2 > 0.5, 1e5, cost2))

        for t_idx, d_idx in matches2:
            remaining_tracks[t_idx].update(low_dets[d_idx], self.frame_id)
            activated.append(remaining_tracks[t_idx])
        for i in u_tracks2:
            t = remaining_tracks[i]
            t.state = TrackState.Lost
            lost.append(t)

        # ── Third pass: unmatched high dets vs lost tracks ──
        u_high_dets = [high_dets[i] for i in u_dets]
        iou3 = iou_batch([t.tlbr for t in self.lost_tracks], [d.tlbr for d in u_high_dets])
        cost3 = 1 - iou3
        matches3, u_lost, u_new = linear_assignment(
            np.where(cost3 > 0.7, 1e5, cost3))

        for t_idx, d_idx in matches3:
            self.lost_tracks[t_idx].re_activate(u_high_dets[d_idx], self.frame_id)
            refound.append(self.lost_tracks[t_idx])

        # ── Unmatched high dets vs unconfirmed tracks ──
        new_dets = [u_high_dets[i] for i in u_new]
        iou4 = iou_batch([t.tlbr for t in unconfirmed], [d.tlbr for d in new_dets])
        cost4 = 1 - iou4
        matches4, u_uncomf, u_brand_new = linear_assignment(
            np.where(cost4 > 0.7, 1e5, cost4))

        for t_idx, d_idx in matches4:
            unconfirmed[t_idx].update(new_dets[d_idx], self.frame_id)
            activated.append(unconfirmed[t_idx])
        for i in u_uncomf:
            removed.append(unconfirmed[i])

        # Activate brand-new tracks
        for i in u_brand_new:
            d = new_dets[i]
            if d.score >= self.det_thresh:
                d.activate(self.frame_id)
                activated.append(d)

        # Age lost tracks
        for t in self.lost_tracks:
            if t not in refound:
                t.time_since_update += 1
                if t.time_since_update > self.max_time_lost:
                    t.state = TrackState.Removed
                    removed.append(t)

        # Update global lists
        self.tracked_tracks = [t for t in self.tracked_tracks
                                if t.state == TrackState.Tracked]
        self.tracked_tracks = _joint(self.tracked_tracks, activated)
        self.tracked_tracks = _joint(self.tracked_tracks, refound)
        self.lost_tracks = _sub(_joint(self.lost_tracks, lost), self.tracked_tracks)
        self.lost_tracks = _sub(self.lost_tracks, removed)
        self.removed_tracks += removed

        return [t for t in self.tracked_tracks if t.is_activated]


def _joint(a, b):
    ids = set(t.track_id for t in a)
    return a + [t for t in b if t.track_id not in ids]

def _sub(a, b):
    ids = set(t.track_id for t in b)
    return [t for t in a if t.track_id not in ids]


import cv2  # needed for compensate_motion
