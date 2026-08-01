"""
reinforcement.py - 强化学习反馈机制
基于管理员反馈动态调整检测阈值和模型权重
"""
import json, os, time
import numpy as np
from datetime import datetime


class ReinforcementLearner:
    """强化学习：根据反馈调整检测参数"""

    def __init__(self, db, save_path="models/rl_params.json"):
        self._db = db
        self._save_path = save_path
        self._params = self._load_params()
        self._feedback_buffer = []

    def _load_params(self):
        if os.path.exists(self._save_path):
            with open(self._save_path, "r") as f:
                return json.load(f)
        return {
            "base_thresholds": {"cigarette": 0.35, "cigarette_pack": 0.35},
            "adaptive_thresholds": {"cigarette": 0.35, "cigarette_pack": 0.35},
            "learning_rate": 0.05,
            "reward_history": [],
            "total_episodes": 0,
        }

    def save_params(self):
        os.makedirs(os.path.dirname(self._save_path), exist_ok=True)
        self._params["total_episodes"] += 1
        with open(self._save_path, "w") as f:
            json.dump(self._params, f, indent=2)

    def get_threshold(self, label):
        """获取自适应阈值"""
        return self._params["adaptive_thresholds"].get(label, 0.35)

    def process_feedback(self, label, was_correct, confidence):
        """处理单条反馈"""
        # 记录到缓冲区
        self._feedback_buffer.append({
            "label": label,
            "was_correct": was_correct,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        })

        # 每收集 5 条反馈调整一次
        if len(self._feedback_buffer) >= 5:
            self._adjust_thresholds()

    def _adjust_thresholds(self):
        """基于反馈调整阈值"""
        lr = self._params["learning_rate"]

        for label in ["cigarette", "cigarette_pack"]:
            label_feedbacks = [f for f in self._feedback_buffer if f["label"] == label]
            if not label_feedbacks:
                continue

            correct = sum(1 for f in label_feedbacks if f["was_correct"])
            total = len(label_feedbacks)
            accuracy = correct / max(total, 1)

            current = self._params["adaptive_thresholds"].get(label, 0.35)
            base = self._params["base_thresholds"].get(label, 0.35)

            if accuracy >= 0.9:
                # 准确率高，稍微降低阈值（更敏感）
                new_threshold = max(0.20, current - lr * 0.5)
            elif accuracy >= 0.7:
                # 准确率一般，微调
                new_threshold = current
            elif accuracy >= 0.5:
                # 准确率偏低，提高阈值
                new_threshold = min(0.60, current + lr)
            else:
                # 准确率很低，大幅提高阈值
                new_threshold = min(0.70, current + lr * 2)

            self._params["adaptive_thresholds"][label] = round(new_threshold, 4)

        # 记录奖励历史
        total_correct = sum(1 for f in self._feedback_buffer if f["was_correct"])
        self._params["reward_history"].append({
            "episode": self._params["total_episodes"],
            "correct": total_correct,
            "total": len(self._feedback_buffer),
            "accuracy": total_correct / max(len(self._feedback_buffer), 1),
        })

        # 只保留最近 100 条历史
        if len(self._params["reward_history"]) > 100:
            self._params["reward_history"] = self._params["reward_history"][-100:]

        self._feedback_buffer = []
        self.save_params()

    def get_stats(self):
        """获取强化学习统计"""
        recent = self._params["reward_history"][-10:] if self._params["reward_history"] else []
        if recent:
            recent_acc = sum(r["accuracy"] for r in recent) / len(recent)
        else:
            recent_acc = 0.0

        return {
            "thresholds": self._params["adaptive_thresholds"],
            "total_episodes": self._params["total_episodes"],
            "recent_accuracy": recent_acc,
            "learning_rate": self._params["learning_rate"],
        }

    def should_retrain(self):
        """判断是否需要重新训练（反馈积累到一定量）"""
        if len(self._params["reward_history"]) >= 10:
            recent_5 = self._params["reward_history"][-5:]
            avg_acc = sum(r["accuracy"] for r in recent_5) / len(recent_5)
            return avg_acc < 0.6
        return False