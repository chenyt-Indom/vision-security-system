"""
detector.py - 多级检测管线
S1: 人体检测 (YOLOv8n) → 绿色框 + 持续追踪
S2: 手部/嘴部 ROI 估算 → 蓝色虚线框
S3: 香烟+烟盒检测 (smoking.onnx) → 红色告警框
"""
import onnxruntime as ort
import numpy as np
import cv2
import time
from typing import List, Dict, Optional, Tuple
from person_tracker import PersonTracker


class MultiStageDetector:
    """多级检测管线"""

    # 人体检测类别 (COCO 80)
    PERSON_CLASS = 0

    # 抽烟检测类别
    SMOKING_CLASSES = {0: "cigarette", 1: "cigarette_pack"}

    def __init__(self, person_model_path: str, smoking_model_path: str,
                 conf_threshold: float = 0.35, iou_threshold: float = 0.45):
        self._conf = conf_threshold
        self._iou_thresh = iou_threshold

        # S1: 人体检测模型
        self._person_session = ort.InferenceSession(
            person_model_path, providers=["CPUExecutionProvider"])
        self._person_input_name = self._person_session.get_inputs()[0].name
        self._person_input_shape = self._person_session.get_inputs()[0].shape
        self._person_img_size = self._person_input_shape[2]

        # S3: 抽烟检测模型
        self._smoke_session = ort.InferenceSession(
            smoking_model_path, providers=["CPUExecutionProvider"])
        self._smoke_input_name = self._smoke_session.get_inputs()[0].name
        self._smoke_input_shape = self._smoke_session.get_inputs()[0].shape
        self._smoke_img_size = self._smoke_input_shape[2]
        self._smoke_num_classes = self._smoke_session.get_outputs()[0].shape[1] - 4

        # 追踪器
        self._tracker = PersonTracker(max_missed=15, iou_thresh=0.3)

        # 性能统计
        self._inference_times: List[float] = []
        self._fps = 0.0
        self._frame_count = 0

    @property
    def avg_inference_time_ms(self) -> float:
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times)

    @property
    def current_fps(self) -> float:
        return self._fps

    @property
    def backend_name(self) -> str:
        return "CPU"

    def detect(self, frame: np.ndarray) -> Dict:
        """
        多级检测主入口
        返回:
        {
            "persons": [{id, bbox, hand_roi, mouth_roi}],  # 人体追踪
            "alerts": [{label, confidence, bbox, track_id}], # 告警
            "rois": [{type, bbox, track_id}],              # ROI区域
        }
        """
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        # === S1: 人体检测 ===
        person_dets = self._detect_persons(frame)

        # === 人体追踪 ===
        tracked = self._tracker.update(person_dets)

        # === S2-S3: 对每个追踪到的人体做ROI检测 ===
        all_alerts = []
        all_rois = []

        for track in tracked:
            # 手部ROI
            hr = track.hand_roi
            if hr:
                hx1, hy1, hx2, hy2 = self._clip_roi(hr, w, h)
                if hx2 - hx1 > 10 and hy2 - hy1 > 10:
                    all_rois.append({"type": "hand", "bbox": [hx1, hy1, hx2, hy2], "track_id": track.id})
                    roi_frame = frame[hy1:hy2, hx1:hx2]
                    smoke_dets = self._detect_smoking(roi_frame)
                    for sd in smoke_dets:
                        # 映射回原图坐标
                        sx1 = int(hx1 + sd["bbox"][0])
                        sy1 = int(hy1 + sd["bbox"][1])
                        sx2 = int(hx1 + sd["bbox"][2])
                        sy2 = int(hy1 + sd["bbox"][3])
                        all_alerts.append({
                            "label": sd["label"],
                            "confidence": sd["confidence"],
                            "bbox": [sx1, sy1, sx2, sy2],
                            "track_id": track.id,
                        })

            # 嘴部ROI
            mr = track.mouth_roi
            if mr:
                mx1, my1, mx2, my2 = self._clip_roi(mr, w, h)
                if mx2 - mx1 > 10 and my2 - my1 > 10:
                    all_rois.append({"type": "mouth", "bbox": [mx1, my1, mx2, my2], "track_id": track.id})
                    roi_frame = frame[my1:my2, mx1:mx2]
                    smoke_dets = self._detect_smoking(roi_frame)
                    for sd in smoke_dets:
                        sx1 = int(mx1 + sd["bbox"][0])
                        sy1 = int(my1 + sd["bbox"][1])
                        sx2 = int(mx1 + sd["bbox"][2])
                        sy2 = int(my1 + sd["bbox"][3])
                        all_alerts.append({
                            "label": sd["label"],
                            "confidence": sd["confidence"],
                            "bbox": [sx1, sy1, sx2, sy2],
                            "track_id": track.id,
                        })

        # 性能统计
        elapsed = (time.perf_counter() - t0) * 1000
        self._inference_times.append(elapsed)
        if len(self._inference_times) > 50:
            self._inference_times = self._inference_times[-50:]
        self._frame_count += 1
        if self._frame_count >= 10:
            self._fps = 1000 / self.avg_inference_time_ms
            self._frame_count = 0

        return {
            "persons": [{"id": t.id, "bbox": t.bbox} for t in tracked],
            "alerts": all_alerts,
            "rois": all_rois,
        }

    def _detect_persons(self, frame: np.ndarray) -> List[List]:
        """使用 YOLOv8n 检测人体"""
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (self._person_img_size, self._person_img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)

        outputs = self._person_session.run(None, {self._person_input_name: img})
        preds = outputs[0]
        if len(preds.shape) == 3:
            preds = np.transpose(preds[0], (1, 0))

        boxes_raw = preds[:, :4]
        scores = preds[:, 4:]

        # 只取 person 类 (class_id=0)
        person_scores = scores[:, 0]
        mask = person_scores > self._conf
        if not np.any(mask):
            return []

        indices = np.where(mask)[0]
        # 按置信度排序
        order = np.argsort(person_scores[indices])[::-1]
        indices = indices[order]

        boxes = boxes_raw[indices]
        confs = person_scores[indices]

        # NMS
        keep = self._nms(boxes, confs)
        indices = indices[keep]

        detections = []
        scale_x = w / self._person_img_size
        scale_y = h / self._person_img_size

        for idx in indices:
            cx, cy, bw, bh = boxes_raw[idx]
            x1 = int((cx - bw / 2) * scale_x)
            y1 = int((cy - bh / 2) * scale_y)
            x2 = int((cx + bw / 2) * scale_x)
            y2 = int((cy + bh / 2) * scale_y)
            detections.append([x1, y1, x2, y2, float(person_scores[idx])])

        return detections

    def _detect_smoking(self, roi_frame: np.ndarray) -> List[Dict]:
        """在 ROI 区域检测香烟和烟盒"""
        if roi_frame.size == 0:
            return []

        h, w = roi_frame.shape[:2]
        img = cv2.resize(roi_frame, (self._smoke_img_size, self._smoke_img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)

        outputs = self._smoke_session.run(None, {self._smoke_input_name: img})
        preds = outputs[0]
        if len(preds.shape) == 3:
            preds = np.transpose(preds[0], (1, 0))

        boxes_raw = preds[:, :4]
        scores = preds[:, 4:]

        if self._smoke_num_classes == 1:
            max_scores = scores[:, 0]
            class_ids = np.zeros(len(max_scores), dtype=np.int32)
        else:
            class_ids = np.argmax(scores, axis=1)
            max_scores = np.max(scores, axis=1)

        mask = max_scores > self._conf
        if not np.any(mask):
            return []

        indices = np.where(mask)[0]
        order = np.argsort(max_scores[indices])[::-1]
        indices = indices[order]

        keep = self._nms(boxes_raw[indices], max_scores[indices])
        indices = indices[keep]

        results = []
        scale_x = w / self._smoke_img_size
        scale_y = h / self._smoke_img_size

        for idx in indices:
            cls_id = int(class_ids[idx])
            label = self.SMOKING_CLASSES.get(cls_id, f"class_{cls_id}")
            cx, cy, bw, bh = boxes_raw[idx]
            x1 = int((cx - bw / 2) * scale_x)
            y1 = int((cy - bh / 2) * scale_y)
            x2 = int((cx + bw / 2) * scale_x)
            y2 = int((cy + bh / 2) * scale_y)
            results.append({
                "label": label,
                "confidence": float(max_scores[idx]),
                "bbox": [x1, y1, x2, y2],
            })

        return results

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        """NMS 非极大值抑制"""
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        areas = (x2 - x1) * (y2 - y1)
        order = np.argsort(scores)[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou < self._iou_thresh]
        return keep

    @staticmethod
    def _clip_roi(roi, img_w, img_h):
        x1, y1, x2, y2 = roi
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(1, min(x2, img_w))
        y2 = max(1, min(y2, img_h))
        return x1, y1, x2, y2