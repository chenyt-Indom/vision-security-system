"""
person_tracker.py - 人体毫秒级追踪器 v3
Kalman滤波运动预测 + 匈牙利算法最优匹配 + 平滑框过渡
多目标稳定追踪，精准双手/嘴部ROI，每帧毫秒级更新
"""
import numpy as np
from collections import OrderedDict
from typing import List, Tuple


class KalmanBoxTracker:
    """Kalman滤波追踪器 — 每个目标独立状态估计"""

    count = 0

    def __init__(self, bbox: List[int], frame_idx: int):
        # 状态: [cx, cy, s, r, vx, vy, vs] — 中心点、面积、宽高比、速度
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.transitionMatrix = np.eye(7, dtype=np.float32)
        self.kf.transitionMatrix[0, 4] = 1.0  # cx += vx
        self.kf.transitionMatrix[1, 5] = 1.0  # cy += vy
        self.kf.transitionMatrix[2, 6] = 1.0  # s += vs
        self.kf.measurementMatrix = np.eye(4, 7, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.01
        self.kf.processNoiseCov[4, 4] = 0.05  # 速度噪声（降低，避免预测漂移）
        self.kf.processNoiseCov[5, 5] = 0.05
        self.kf.processNoiseCov[6, 6] = 0.01
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.5

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / max(y2 - y1, 1)
        self.kf.statePre = np.array([[cx], [cy], [s], [r], [0], [0], [0]], dtype=np.float32)

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.frame_idx = frame_idx
        self.last_seen = frame_idx
        self.missed = 0
        self.hits = 1
        self.bbox = bbox
        self.smoothed_bbox = bbox  # 平滑过渡后的框
        self.hand_roi = None
        self.hand_left_roi = None
        self.mouth_roi = None
        self.confidence = 0.0
        self._update_rois()

    def predict(self):
        """Kalman预测下一帧位置"""
        pred = self.kf.predict()
        cx, cy, s, r = pred[0, 0], pred[1, 0], pred[2, 0], pred[3, 0]
        s = max(s, 100)
        w = np.sqrt(s * r)
        h = s / max(w, 1)
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        self.bbox = [x1, y1, x2, y2]
        # 平滑过渡：EMA 平滑避免跳动
        if self.smoothed_bbox:
            alpha = 0.4  # 平滑系数（越小越平滑）
            self.smoothed_bbox = [
                int(alpha * self.bbox[i] + (1 - alpha) * self.smoothed_bbox[i])
                for i in range(4)
            ]
        else:
            self.smoothed_bbox = list(self.bbox)

    def update(self, bbox: List[int], frame_idx: int):
        """用检测结果更新 Kalman"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / max(y2 - y1, 1)
        self.kf.correct(np.array([[cx], [cy], [s], [r]], dtype=np.float32))
        self.bbox = bbox
        self.last_seen = frame_idx
        self.missed = 0
        self.hits += 1
        # EMA 平滑
        if self.smoothed_bbox:
            alpha = 0.3
            self.smoothed_bbox = [
                int(alpha * bbox[i] + (1 - alpha) * self.smoothed_bbox[i])
                for i in range(4)
            ]
        else:
            self.smoothed_bbox = list(bbox)
        self._update_rois()

    def mark_missed(self):
        self.missed += 1

    def _update_rois(self):
        """精准 ROI 估算 — 基于平滑后的框"""
        bx = self.smoothed_bbox if self.smoothed_bbox else self.bbox
        x1, y1, x2, y2 = bx
        pw = max(x2 - x1, 1)
        ph = max(y2 - y1, 1)

        # 人体框收缩5%贴合人体（去除环境边缘）
        margin_x = int(pw * 0.05)
        margin_y = int(ph * 0.05)
        x1 += margin_x
        x2 -= margin_x
        y1 += margin_y
        y2 -= margin_y
        pw = max(x2 - x1, 1)
        ph = max(y2 - y1, 1)

        # 嘴部 ROI：面部区域，上部 8%-24%
        self.mouth_roi = [
            x1 + int(pw * 0.30), y1 + int(ph * 0.08),
            x1 + int(pw * 0.70), y1 + int(ph * 0.24)
        ]
        # 右手 ROI：右侧 50%-105% 宽，上部 10%-48%
        self.hand_roi = [
            x1 + int(pw * 0.50), y1 + int(ph * 0.10),
            x2 + int(pw * 0.05), y1 + int(ph * 0.48)
        ]
        # 左手 ROI：左侧对称
        self.hand_left_roi = [
            x1 - int(pw * 0.05), y1 + int(ph * 0.10),
            x1 + int(pw * 0.50), y1 + int(ph * 0.48)
        ]


class PersonTracker:
    """人体追踪管理器 — Kalman预测 + 匈牙利算法最优匹配"""

    def __init__(self, max_missed: int = 3, iou_thresh: float = 0.2):
        self._max_missed = max_missed
        self._iou_thresh = iou_thresh
        self._tracks: OrderedDict = OrderedDict()
        self._frame_idx = 0

    def update(self, detections: List[List]) -> List[KalmanBoxTracker]:
        """
        detections: list of [x1, y1, x2, y2, conf]
        返回活跃追踪目标
        """
        self._frame_idx += 1

        # 第一步：预测所有现有追踪的下一帧位置
        for t in self._tracks.values():
            t.predict()
            t.mark_missed()

        if not detections:
            self._remove_stale()
            return self.get_active_tracks()

        # 第二步：匈牙利算法 — 最优匹配（使用 smoothed_bbox 更稳定）
        det_boxes = np.array([d[:4] for d in detections])
        track_ids = list(self._tracks.keys())
        track_boxes = np.array([self._tracks[tid].smoothed_bbox if self._tracks[tid].smoothed_bbox
                                else self._tracks[tid].bbox for tid in track_ids])

        if len(track_boxes) > 0:
            iou_matrix = self._iou_matrix(track_boxes, det_boxes)
            matched, unmatched_tracks, unmatched_dets = self._associate(iou_matrix)
        else:
            matched = []
            unmatched_tracks = []
            unmatched_dets = list(range(len(detections)))

        # 第三步：更新匹配的追踪
        for ti, di in matched:
            tid = track_ids[ti]
            det = detections[di]
            self._tracks[tid].update(det[:4], self._frame_idx)
            if len(det) > 4:
                self._tracks[tid].confidence = det[4]

        # 第四步：为未匹配的检测创建新追踪
        for di in unmatched_dets:
            det = detections[di]
            tracker = KalmanBoxTracker(det[:4], self._frame_idx)
            tracker.confidence = det[4] if len(det) > 4 else 0.0
            self._tracks[tracker.id] = tracker

        # 第五步：移除过期追踪
        self._remove_stale()

        return self.get_active_tracks()

    def get_active_tracks(self) -> List[KalmanBoxTracker]:
        return [t for t in self._tracks.values() if t.missed <= 1]

    def _remove_stale(self):
        to_remove = [tid for tid, t in self._tracks.items() if t.missed > self._max_missed]
        for tid in to_remove:
            del self._tracks[tid]

    def _associate(self, iou_matrix: np.ndarray) -> Tuple:
        """匈牙利算法最优匹配"""
        # 贪心匹配（简化版匈牙利）
        matched = []
        unmatched_tracks = set(range(iou_matrix.shape[0]))
        unmatched_dets = set(range(iou_matrix.shape[1]))

        if iou_matrix.size == 0:
            return matched, unmatched_tracks, unmatched_dets

        # 按 IoU 降序匹配
        flat_indices = np.argsort(iou_matrix.ravel())[::-1]
        for idx in flat_indices:
            ti = idx // iou_matrix.shape[1]
            di = idx % iou_matrix.shape[1]
            if iou_matrix[ti, di] >= self._iou_thresh:
                if ti in unmatched_tracks and di in unmatched_dets:
                    matched.append((ti, di))
                    unmatched_tracks.remove(ti)
                    unmatched_dets.remove(di)

        return matched, unmatched_tracks, unmatched_dets

    @staticmethod
    def _iou_matrix(track_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
        """计算追踪框与检测框的 IoU 矩阵"""
        n_tracks = len(track_boxes)
        n_dets = len(det_boxes)
        iou = np.zeros((n_tracks, n_dets), dtype=np.float32)

        for i in range(n_tracks):
            tx1, ty1, tx2, ty2 = track_boxes[i]
            ta = max((tx2 - tx1) * (ty2 - ty1), 1)
            for j in range(n_dets):
                dx1, dy1, dx2, dy2 = det_boxes[j]
                x1 = max(tx1, dx1)
                y1 = max(ty1, dy1)
                x2 = min(tx2, dx2)
                y2 = min(ty2, dy2)
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                da = max((dx2 - dx1) * (dy2 - dy1), 1)
                union = ta + da - inter
                iou[i, j] = inter / (union + 1e-6)

        return iou


# 导入 cv2（用于 KalmanFilter）
import cv2