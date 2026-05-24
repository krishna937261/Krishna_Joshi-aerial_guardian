"""
TrajectoryBuffer — stores centroid history per track ID.

The tail is displayed as a short polyline behind each bounding box.
An EMA (Exponential Moving Average) smoother is applied to the stored
positions so that camera shake does not make tails appear jittery.
"""

from collections import defaultdict, deque
from typing import Dict, List, Tuple
import numpy as np


class TrajectoryBuffer:
    def __init__(self, max_len: int = 30, ema_alpha: float = 0.4):
        """
        max_len   : how many past centres to keep per track
        ema_alpha : smoothing factor (0 = fully smooth, 1 = no smoothing)
                    0.4 is a good balance for 25-30 fps drone footage
        """
        self.max_len = max_len
        self.alpha = ema_alpha
        # Raw history: track_id → deque of (cx, cy)
        self._raw: Dict[int, deque] = defaultdict(lambda: deque(maxlen=max_len))
        # EMA-smoothed last position
        self._ema: Dict[int, Tuple[float, float]] = {}

    def update(self, track_id: int, cx: float, cy: float):
        if track_id not in self._ema:
            self._ema[track_id] = (cx, cy)
        else:
            ex, ey = self._ema[track_id]
            self._ema[track_id] = (
                self.alpha * cx + (1 - self.alpha) * ex,
                self.alpha * cy + (1 - self.alpha) * ey,
            )
        self._raw[track_id].append(self._ema[track_id])

    def get(self, track_id: int) -> List[Tuple[int, int]]:
        return [(int(x), int(y)) for x, y in self._raw.get(track_id, [])]

    def remove(self, track_id: int):
        self._raw.pop(track_id, None)
        self._ema.pop(track_id, None)

    def prune(self, active_ids):
        """Remove buffers for tracks no longer active."""
        stale = set(self._raw.keys()) - set(active_ids)
        for tid in stale:
            self.remove(tid)
