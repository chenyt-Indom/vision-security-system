"""
detector.py - 极速多级检测管线 v2
S1: 人体检测 (YOLOv8n+ONNX图优化) → 绿色框
S2: 手部/嘴部 ROI 精准估算 → 蓝色框 + 跟踪显示
S3: 香烟+烟盒检测 (smoking.onnx) → 红色/橙色告警框
优化: ONNX图优化 + 预分配数组 + blobFromImage + 帧跳过 + ROI缓存
"""
import onnxruntime as ort
import numpy as np
import cv2
import time
from typing import List, Dict, Optional, Tuple
from person_tracker import PersonTracker


class MultiStageDetector:
    """极速多级检测管线"""

    PERSON_CLASS = 0
    SMOKING_CLASSES = {0: "cigarette", 1: "cigarette_pack"}

    def __init__(self, person_model_path: str, smoking_model_path: str,
                 conf_threshold: float = 0.35, iou_threshold: float = 0.45):
        self._conf = conf_threshold
        self._iou_thresh = iou_threshold

        # ONNX 会话选项：图优化 + 线程绑定 + 最快推理
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True

        # S1: 人体检测模型
        self._person_session = ort.InferenceSession(
            person_model_path, opts, providers=["CPUExecutionProvider"])
        self._person_input_name = self._person_session.get_inputs()[0].name
        self._person_img_size = self._person_session.get_inputs()[0].shape[2]

        # S3: 抽烟检测模型
        self._smoke_session = ort.InferenceSession(
            smoking_model_path, opts, providers=["CPUExecutionProvider"])
        self._smoke_input_name = self._smoke_session.get_inputs()[0].name
        self._smoke_img_size = self._smoke_session.get_inputs()[0].shape[2]
        smoke_out_shape = self._smoke_session.get_outputs()[0].shape
        self._smoke_num_classes = smoke_out_shape[1] - 4 if len(smoke_out_shape) >= 2 else 1

        # 追踪器
        self._tracker = PersonTracker(max_missed=15, iou_thresh=0.3)

        # 预分配预处理数组（避免每帧分配）
        self._person_blob = np.zeros((1, 3, self._person_img_size, self._person_img_size), dtype=np.float32)
        self._smoke_blob = np.zeros((1, 3, self._smoke_img_size, self._smoke_img_size), dtype=np.float32)

        # 帧跳过：每人物每 N 帧做一次抽烟检测
        self._person_frame_counters = {}  # track_id -> frame_count
        self._skip_interval = 2           # 每2帧做一次抽烟检测
        self._last_roi_cache = {}         # track_id -> (hand_roi, mouth_roi, hash)

        # 性能统计
        self._inference_times: List[float] = []
        self._fps = 0.0
        self._frame_count = 0
        self._global_frame = 0

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
        return "CPU(ONNX优化)"

    def detect(self, frame: np.ndarray) -> Dict:
        """极速多级检测"""
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        self._global_frame += 1

        # === S1: 人体检测（blobFromImage 加速预处理）===
        person_dets = self._detect_persons_fast(frame)

        # === 人体追踪 ===
        tracked = self._tracker.update(person_dets)

        # === S2-S3: 精准 ROI 检测 + 帧跳过 ===
        all_alerts = []
        all_rois = []

        for track in tracked:
            tid = track.id
            if tid not in self._person_frame_counters:
                self._person_frame_counters[tid] = 0
            self._person_frame_counters[tid] += 1
            do_smoke = self._person_frame_counters[tid] % self._skip_interval == 0

            # 嘴部 ROI
            mr = track.mouth_roi
            if mr:
                mx1, my1, mx2, my2 = self._clip_roi(mr, w, h)
                all_rois.append({"type": "mouth", "bbox": [mx1, my1, mx2, my2], "track_id": tid})
                if do_smoke and my2 > my1 and mx2 > mx1:
                    smoke_dets = self._detect_smoking_fast(frame[my1:my2, mx1:mx2])
                    for sd in smoke_dets:
                        all_alerts.append({
                            "label": sd["label"], "confidence": sd["confidence"],
                            "bbox": [int(mx1 + sd["bbox"][0]), int(my1 + sd["bbox"][1]),
                                     int(mx1 + sd["bbox"][2]), int(my1 + sd["bbox"][3])],
                            "track_id": tid,
                        })

            # 右手 ROI
            hr = track.hand_roi
            if hr:
                hx1, hy1, hx2, hy2 = self._clip_roi(hr, w, h)
                all_rois.append({"type": "hand_r", "bbox": [hx1, hy1, hx2, hy2], "track_id": tid})
                if do_smoke and hy2 > hy1 and hx2 > hx1:
                    smoke_dets = self._detect_smoking_fast(frame[hy1:hy2, hx1:hx2])
                    for sd in smoke_dets:
                        all_alerts.append({
                            "label": sd["label"], "confidence": sd["confidence"],
                            "bbox": [int(hx1 + sd["bbox"][0]), int(hy1 + sd["bbox"][1]),
                                     int(hx1 + sd["bbox"][2]), int(hy1 + sd["bbox"][3])],
                            "track_id": tid,
                        })

            # 左手 ROI
            lr = track.hand_left_roi
            if lr:
                lx1, ly1, lx2, ly2 = self._clip_roi(lr, w, h)
                all_rois.append({"type": "hand_l", "bbox": [lx1, ly1, lx2, ly2], "track_id": tid})
                if do_smoke and ly2 > ly1 and lx2 > lx1:
                    smoke_dets = self._detect_smoking_fast(frame[ly1:ly2, lx1:lx2])
                    for sd in smoke_dets:
                        all_alerts.append({
                            "label": sd["label"], "confidence": sd["confidence"],
                            "bbox": [int(lx1 + sd["bbox"][0]), int(ly1 + sd["bbox"][1]),
                                     int(lx1 + sd["bbox"][2]), int(ly1 + sd["bbox"][3])],
                            "track_id": tid,
                        })

        # 清理不活跃的计数器
        active_ids = {t.id for t in tracked}
        self._person_frame_counters = {tid: cnt for tid, cnt in self._person_frame_counters.items() if tid in active_ids}

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
            "persons": [{"id": t.id, "bbox": t.bbox,
                         "hand_roi": t.hand_roi, "hand_left_roi": t.hand_left_roi, "mouth_roi": t.mouth_roi}
                        for t in tracked],
            "alerts": all_alerts,
            "rois": all_rois,
        }

    def _detect_persons_fast(self, frame: np.ndarray) -> List[List]:
        """人体检测 - blobFromImage 加速预处理"""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (self._person_img_size, self._person_img_size),
                                     swapRB=True, crop=False)
        outputs = self._person_session.run(None, {self._person_input_name: blob})
        preds = outputs[0]
        if len(preds.shape) == 3:
            preds = np.transpose(preds[0], (1, 0))

        boxes_raw = preds[:, :4]
        person_scores = preds[:, 4]

        mask = person_scores > self._conf
        if not np.any(mask):
            return []

        indices = np.where(mask)[0]
        order = np.argsort(person_scores[indices])[::-1]
        indices = indices[order]

        keep = self._nms(boxes_raw[indices], person_scores[indices])
        indices = indices[keep]

        scale_x = w / self._person_img_size
        scale_y = h / self._person_img_size

        detections = []
        for idx in indices:
            cx, cy, bw, bh = boxes_raw[idx]
            x1 = int((cx - bw / 2) * scale_x)
            y1 = int((cy - bh / 2) * scale_y)
            x2 = int((cx + bw / 2) * scale_x)
            y2 = int((cy + bh / 2) * scale_y)
            detections.append([x1, y1, x2, y2, float(person_scores[idx])])

        return detections

    def _detect_smoking_fast(self, roi_frame: np.ndarray) -> List[Dict]:
        """抽烟检测 - blobFromImage 加速"""
        if roi_frame.size == 0:
            return []

        rh, rw = roi_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(roi_frame, 1/255.0, (self._smoke_img_size, self._smoke_img_size),
                                     swapRB=True, crop=False)
        outputs = self._smoke_session.run(None, {self._smoke_input_name: blob})
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

        scale_x = rw / self._smoke_img_size
        scale_y = rh / self._smoke_img_size

        results = []
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