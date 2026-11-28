"""Multi-frame unique object tracker for Stage 2 jewellery counting.

Assigns persistent unique IDs (e.g., `JWL-001`, `JWL-002`) to individual
jewellery articles on the testbed, prevents duplicate counting across frames,
and accurately handles sequential placement as well as removal and re-placement
by maintaining active on-testbed track state instead of naive cumulative counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.image_utils import bbox_iou, mask_iou
from ..utils.inspection_config import InspectionConfig
from .instance_separator import SeparatedArticle


def _linear_assignment(cost_matrix: np.ndarray) -> Tuple[List[int], List[int]]:
    """Solve minimum-cost bipartite matching (uses scipy when available, greedy fallback)."""
    if cost_matrix.size == 0:
        return [], []
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        rows, cols = linear_sum_assignment(cost_matrix)
        return rows.tolist(), cols.tolist()
    except Exception:
        # Deterministic greedy fallback
        rows_out: List[int] = []
        cols_out: List[int] = []
        used_r: set[int] = set()
        used_c: set[int] = set()
        flat_indices = np.argsort(cost_matrix, axis=None)
        n_cols = cost_matrix.shape[1]
        for flat_idx in flat_indices:
            r = int(flat_idx // n_cols)
            c = int(flat_idx % n_cols)
            if r not in used_r and c not in used_c:
                used_r.add(r)
                used_c.add(c)
                rows_out.append(r)
                cols_out.append(c)
        return rows_out, cols_out


@dataclass
class TrackedJewel:
    track_id: str
    numeric_id: int
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    area: int
    confidence: float
    hits: int = 1
    missing_frames: int = 0
    separated_from_cluster: bool = False
    uncertain: bool = False
    last_centroid_shift: float = 0.0
    last_area_change_ratio: float = 0.0
    confirmed: bool = False


@dataclass
class TrackingSnapshot:
    active_tracks: List[TrackedJewel]
    confirmed_tracks: List[TrackedJewel]
    live_count: int
    newly_added_ids: List[str] = field(default_factory=list)
    removed_ids: List[str] = field(default_factory=list)


class JewelleryObjectTracker:
    """Associates separated jewellery articles across video frames with unique IDs."""

    def __init__(self, config: InspectionConfig) -> None:
        self.config = config
        self.trk_cfg = config.jewellery_counting.tracking
        self._next_id: int = 1
        self._tracks: Dict[int, TrackedJewel] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()

    def _allocate_id(self) -> Tuple[int, str]:
        num = self._next_id
        self._next_id += 1
        return num, f"{self.trk_cfg.id_prefix}-{num:03d}"

    def _pair_score(self, track: TrackedJewel, art: SeparatedArticle) -> float:
        """Higher score = better match in [0, 1]; returns -1 if infeasible."""
        dist = float(np.hypot(track.centroid[0] - art.centroid[0], track.centroid[1] - art.centroid[1]))
        max_dist = float(self.trk_cfg.max_centroid_distance_px)
        m_iou = mask_iou(track.mask, art.mask)
        b_iou = bbox_iou(track.bbox, art.bbox)

        if m_iou < self.trk_cfg.min_match_iou and b_iou < self.trk_cfg.min_match_iou and dist > max_dist:
            return -1.0

        dist_score = max(0.0, 1.0 - (dist / max(max_dist, 1.0)))
        return 0.60 * m_iou + 0.20 * b_iou + 0.20 * dist_score

    def update(self, articles: List[SeparatedArticle]) -> TrackingSnapshot:
        """Update active tracks with current-frame separated jewellery articles."""
        track_keys = list(self._tracks.keys())
        matched_tracks: set[int] = set()
        matched_articles: set[int] = set()
        newly_added: List[str] = []
        removed: List[str] = []

        if track_keys and articles:
            cost = np.full((len(track_keys), len(articles)), 1e6, dtype=np.float64)
            for ti, tkey in enumerate(track_keys):
                trk = self._tracks[tkey]
                for ai, art in enumerate(articles):
                    score = self._pair_score(trk, art)
                    if score >= 0.0:
                        cost[ti, ai] = 1.0 - score

            rows, cols = _linear_assignment(cost)
            for ti, ai in zip(rows, cols):
                if cost[ti, ai] >= 1e5:
                    continue
                tkey = track_keys[ti]
                trk = self._tracks[tkey]
                art = articles[ai]

                shift = float(np.hypot(trk.centroid[0] - art.centroid[0], trk.centroid[1] - art.centroid[1]))
                area_change = abs(float(art.area) - float(trk.area)) / float(max(trk.area, 1))

                trk.mask = art.mask
                trk.bbox = art.bbox
                trk.centroid = art.centroid
                trk.area = art.area
                trk.confidence = art.confidence
                trk.hits += 1
                trk.missing_frames = 0
                trk.separated_from_cluster = art.separated_from_cluster
                trk.uncertain = art.uncertain
                trk.last_centroid_shift = shift
                trk.last_area_change_ratio = area_change
                if trk.hits >= self.trk_cfg.min_track_confirm_hits:
                    trk.confirmed = True

                matched_tracks.add(tkey)
                matched_articles.add(ai)

        # Increment missing counter for unmatched existing tracks; expire removed items
        for tkey in list(self._tracks.keys()):
            if tkey not in matched_tracks:
                trk = self._tracks[tkey]
                trk.missing_frames += 1
                trk.last_centroid_shift = 0.0
                trk.last_area_change_ratio = 0.0
                if trk.missing_frames > self.trk_cfg.max_missing_frames:
                    removed.append(trk.track_id)
                    del self._tracks[tkey]

        # Spawn new tracks for unmatched articles
        for ai, art in enumerate(articles):
            if ai in matched_articles:
                continue
            num_id, str_id = self._allocate_id()
            confirmed = self.trk_cfg.min_track_confirm_hits <= 1
            self._tracks[num_id] = TrackedJewel(
                track_id=str_id,
                numeric_id=num_id,
                mask=art.mask,
                bbox=art.bbox,
                centroid=art.centroid,
                area=art.area,
                confidence=art.confidence,
                hits=1,
                missing_frames=0,
                separated_from_cluster=art.separated_from_cluster,
                uncertain=art.uncertain,
                confirmed=confirmed,
            )
            newly_added.append(str_id)

        # Active tracks currently visible (missing_frames == 0 or within brief flicker grace)
        active = sorted(self._tracks.values(), key=lambda t: t.numeric_id)
        confirmed_active = [t for t in active if t.confirmed and t.missing_frames <= 1]

        return TrackingSnapshot(
            active_tracks=active,
            confirmed_tracks=confirmed_active,
            live_count=len(confirmed_active),
            newly_added_ids=newly_added,
            removed_ids=removed,
        )

