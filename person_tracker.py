"""person_tracker.py - 人体持续追踪器 (IoU匹配 + Kalman滤波)"""
import numpy as np
from collections import OrderedDict


class TrackedPerson:
    def __init__(self, track_id, bbox, frame_idx):
        self.id = track_id
        self.bbox = bbox        # [x1, y1, x2, y2]
        self.frame_idx = frame_idx
        self.last_seen = frame_idx
        self.missed = 0
        # 手部和嘴部 ROI（基于人体比例估算）
        self.hand_roi = None    # [x1, y1, x2, y2]
        self.mouth_roi = None   # [x1, y1, x2, y2]
        self._update_rois()

    def update(self, bbox, frame_idx):
        self.bbox = bbox
        self.last_seen = frame_idx
        self.missed = 0
        self._update_rois()

    def mark_missed(self):
        self.missed += 1

    def _update_rois(self):
        x1, y1, x2, y2 = self.bbox
        pw = x2 - x1
        ph = y2 - y1

        # 嘴部ROI：人体上部 1/5 到 1/3 区域
        mx1 = x1 + int(pw * 0.30)
        my1 = y1 + int(ph * 0.08)
        mx2 = x1 + int(pw * 0.70)
        my2 = y1 + int(ph * 0.28)
        self.mouth_roi = [mx1, my1, mx2, my2]

        # 手部ROI：人体中部偏右侧区域（抽烟时手通常在嘴附近）
        hx1 = x1 + int(pw * 0.55)
        hy1 = y1 + int(ph * 0.08)
        hx2 = x2 + int(pw * 0.15)
        hy2 = y1 + int(ph * 0.50)
        self.hand_roi = [hx1, hy1, hx2, hy2]


class PersonTracker:
    def __init__(self, max_missed=15, iou_thresh=0.3):
        self._max_missed = max_missed
        self._iou_thresh = iou_thresh
        self._tracks = OrderedDict()
        self._next_id = 0
        self._frame_idx = 0

    def update(self, detections):
        """detections: list of [x1, y1, x2, y2, conf]"""
        self._frame_idx += 1

        # 标记所有现有 track 为未匹配
        for t in self._tracks.values():
            t.mark_missed()

        matched = set()
        if detections:
            det_boxes = np.array([d[:4] for d in detections])
            for tid, track in list(self._tracks.items()):
                ious = self._iou_batch(track.bbox, det_boxes)
                best = np.argmax(ious)
                if ious[best] >= self._iou_thresh and best not in matched:
                    track.update(detections[best][:4], self._frame_idx)
                    matched.add(best)
                    if len(detections[best]) > 4:
                        track.confidence = detections[best][4]

        # 为新检测创建 track
        if detections:
            for i, det in enumerate(detections):
                if i not in matched:
                    self._tracks[self._next_id] = TrackedPerson(
                        self._next_id, det[:4], self._frame_idx)
                    self._next_id += 1

        # 移除丢失太久的 track
        to_remove = [tid for tid, t in self._tracks.items()
                     if t.missed > self._max_missed]
        for tid in to_remove:
            del self._tracks[tid]

        return self.get_active_tracks()

    def get_active_tracks(self):
        return [t for t in self._tracks.values() if t.missed == 0]

    @staticmethod
    def _iou_batch(box, boxes):
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_a = (box[2] - box[0]) * (box[3] - box[1])
        area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_a + area_b - inter
        return inter / (union + 1e-6)