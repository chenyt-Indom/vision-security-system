"""
decision_engine.py - 决策链引擎
连续帧确认 + 滑动时间窗口 + 冷却机制
防止误报，确保安防级可靠性
"""
from collections import defaultdict
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class DecisionEngine:
    """抽烟告警决策引擎"""

    def __init__(self, config: dict):
        self._confirm = config["confirm_frames"]          # 连续确认帧数
        self._window = config["confirm_window_seconds"]   # 时间窗口
        self._cooldown = config["cooldown_seconds"]       # 告警冷却
        self._history: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)
        self._last_alert: Dict[str, float] = {}

    def evaluate(self, cam_id: str, detections: List[Dict]) -> Tuple[bool, Optional[Dict]]:
        """
        评估是否触发告警
        返回: (是否告警, 告警详情)
        """
        now = time.time()
        has_detection = len(detections) > 0

        # 记录当前帧的检测结果
        self._history[cam_id].append((now, has_detection))

        # 清理过期记录
        cutoff = now - self._window
        self._history[cam_id] = [(t, d) for t, d in self._history[cam_id] if t > cutoff]

        # 当前帧无检测，不触发
        if not has_detection:
            return False, None

        # 统计窗口内阳性帧数
        positive = sum(1 for _, d in self._history[cam_id] if d)
        if positive < self._confirm:
            return False, None

        # 冷却检查
        last = self._last_alert.get(cam_id, 0)
        if now - last < self._cooldown:
            return False, None

        # 触发告警
        self._last_alert[cam_id] = now
        best = max(detections, key=lambda d: d["confidence"])
        return True, {
            "cam_id": cam_id,
            "confidence": best["confidence"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "positive_frames": positive,
        }

    def reset(self, cam_id: str = None):
        """重置决策状态"""
        if cam_id:
            self._history.pop(cam_id, None)
            self._last_alert.pop(cam_id, None)
        else:
            self._history.clear()
            self._last_alert.clear()