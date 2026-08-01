"""
database.py - 本地 SQLite 数据库
存储截图证据、管理员反馈、模型评分、强化学习数据
"""
import sqlite3, os, json, time
from datetime import datetime
from threading import Lock


class FeedbackDB:
    """本地反馈数据库"""

    def __init__(self, db_path="alerts/feedback.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._lock = Lock()
        self._init_tables()

    def _init_tables(self):
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()

            # 截图证据表
            c.execute("""
                CREATE TABLE IF NOT EXISTS evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    camera_name TEXT,
                    person_id TEXT,
                    image_path TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox TEXT,
                    person_features TEXT,
                    timestamp TEXT NOT NULL,
                    admin_judgment TEXT DEFAULT 'pending',
                    admin_score INTEGER DEFAULT 0,
                    admin_note TEXT,
                    judged_at TEXT,
                    created_at TEXT DEFAULT (datetime('now','localtime'))
                )
            """)

            # 强化学习反馈表
            c.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evidence_id INTEGER,
                    camera_id TEXT,
                    label TEXT,
                    confidence REAL,
                    was_correct INTEGER,
                    created_at TEXT DEFAULT (datetime('now','localtime')),
                    FOREIGN KEY (evidence_id) REFERENCES evidence(id)
                )
            """)

            # 模型评分表
            c.execute("""
                CREATE TABLE IF NOT EXISTS model_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT NOT NULL,
                    total_correct INTEGER DEFAULT 0,
                    total_incorrect INTEGER DEFAULT 0,
                    accuracy REAL DEFAULT 0.0,
                    confidence_threshold REAL DEFAULT 0.35,
                    updated_at TEXT DEFAULT (datetime('now','localtime'))
                )
            """)

            # 人物特征缓存表
            c.execute("""
                CREATE TABLE IF NOT EXISTS person_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL,
                    camera_id TEXT,
                    features TEXT NOT NULL,
                    last_seen TEXT DEFAULT (datetime('now','localtime')),
                    detection_count INTEGER DEFAULT 1
                )
            """)

            conn.commit()
            conn.close()

    def add_evidence(self, camera_id, camera_name, person_id, image_path,
                     label, confidence, bbox, person_features):
        """添加截图证据"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()
            c.execute("""
                INSERT INTO evidence (camera_id, camera_name, person_id, image_path,
                    label, confidence, bbox, person_features, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (camera_id, camera_name, person_id, image_path, label,
                  confidence, json.dumps(bbox), json.dumps(person_features),
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            eid = c.lastrowid
            conn.commit()
            conn.close()
            return eid

    def judge_evidence(self, evidence_id, is_correct, note=""):
        """管理员判断：正确=+1分，错误=-1分"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()

            judgment = "correct" if is_correct else "incorrect"
            score = 1 if is_correct else -1
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            c.execute("""
                UPDATE evidence SET admin_judgment=?, admin_score=?, admin_note=?, judged_at=?
                WHERE id=?
            """, (judgment, score, note, now, evidence_id))

            # 获取证据信息
            c.execute("SELECT label, confidence, camera_id FROM evidence WHERE id=?", (evidence_id,))
            row = c.fetchone()
            if row:
                label, confidence, cam_id = row
                # 添加反馈记录
                c.execute("""
                    INSERT INTO feedback (evidence_id, camera_id, label, confidence, was_correct)
                    VALUES (?, ?, ?, ?, ?)
                """, (evidence_id, cam_id, label, confidence, 1 if is_correct else 0))

                # 更新模型评分
                self._update_model_score(c, label, is_correct)

            conn.commit()
            conn.close()

    def _update_model_score(self, cursor, label, is_correct):
        """更新模型评分"""
        cursor.execute("SELECT * FROM model_scores WHERE label=?", (label,))
        row = cursor.fetchone()
        if row:
            if is_correct:
                cursor.execute("""
                    UPDATE model_scores SET total_correct=total_correct+1,
                    accuracy=CAST(total_correct+1 AS REAL)/(total_correct+total_incorrect+1),
                    updated_at=datetime('now','localtime')
                    WHERE label=?
                """, (label,))
            else:
                cursor.execute("""
                    UPDATE model_scores SET total_incorrect=total_incorrect+1,
                    accuracy=CAST(total_correct AS REAL)/(total_correct+total_incorrect+1),
                    updated_at=datetime('now','localtime')
                    WHERE label=?
                """, (label,))
        else:
            acc = 1.0 if is_correct else 0.0
            tc = 1 if is_correct else 0
            ti = 0 if is_correct else 1
            cursor.execute("""
                INSERT INTO model_scores (label, total_correct, total_incorrect, accuracy)
                VALUES (?, ?, ?, ?)
            """, (label, tc, ti, acc))

    def get_confidence_threshold(self, label):
        """根据模型评分动态调整置信度阈值"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()
            c.execute("SELECT accuracy, confidence_threshold FROM model_scores WHERE label=?", (label,))
            row = c.fetchone()
            conn.close()
            if row:
                accuracy, threshold = row
                # 准确率高→降低阈值，准确率低→提高阈值
                if accuracy >= 0.9:
                    return max(0.25, threshold - 0.02)
                elif accuracy >= 0.8:
                    return threshold
                elif accuracy >= 0.6:
                    return min(0.55, threshold + 0.03)
                else:
                    return min(0.65, threshold + 0.05)
            return 0.35

    def get_pending_evidence(self, limit=50):
        """获取待审核的截图"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("""
                SELECT * FROM evidence WHERE admin_judgment='pending'
                ORDER BY created_at DESC LIMIT ?
            """, (limit,))
            rows = [dict(r) for r in c.fetchall()]
            conn.close()
            return rows

    def get_recent_judged(self, limit=20):
        """获取最近已审核的记录"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("""
                SELECT * FROM evidence WHERE admin_judgment!='pending'
                ORDER BY judged_at DESC LIMIT ?
            """, (limit,))
            rows = [dict(r) for r in c.fetchall()]
            conn.close()
            return rows

    def get_model_stats(self):
        """获取模型统计信息"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM model_scores")
            rows = [dict(r) for r in c.fetchall()]
            conn.close()
            return rows

    def get_total_stats(self):
        """获取总体统计"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM evidence WHERE admin_judgment='correct'")
            correct = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM evidence WHERE admin_judgment='incorrect'")
            incorrect = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM evidence WHERE admin_judgment='pending'")
            pending = c.fetchone()[0]
            conn.close()
            total = correct + incorrect
            acc = correct / max(total, 1)
            return {"correct": correct, "incorrect": incorrect, "pending": pending,
                    "accuracy": acc, "total": total}

    def find_recent_similar_person(self, camera_id, person_features, time_window=30):
        """查找近期相似人物（用于去重）"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()
            cutoff = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # 查询时间窗口内的记录
            c.execute("""
                SELECT id, person_features, confidence, created_at FROM evidence
                WHERE camera_id=? AND admin_judgment='pending'
                ORDER BY created_at DESC LIMIT 20
            """, (camera_id,))
            rows = c.fetchall()
            conn.close()

            for eid, stored_features, conf, created_at in rows:
                if stored_features:
                    stored = json.loads(stored_features)
                    similarity = self._feature_similarity(person_features, stored)
                    if similarity > 0.75:
                        return eid, conf
            return None, 0

    @staticmethod
    def _feature_similarity(f1, f2):
        """计算两个特征向量的余弦相似度"""
        if not f1 or not f2:
            return 0.0
        import numpy as np
        a = np.array(list(f1.values()))
        b = np.array(list(f2.values()))
        dot = np.dot(a, b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def replace_evidence_image(self, evidence_id, new_image_path, new_confidence):
        """替换已有证据的截图（用更高置信度的替换）"""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()
            c.execute("""
                UPDATE evidence SET image_path=?, confidence=?
                WHERE id=?
            """, (new_image_path, new_confidence, evidence_id))
            conn.commit()
            conn.close()

    def get_stats_for_display(self):
        """获取用于前端显示的统计"""
        stats = self.get_total_stats()
        scores = self.get_model_stats()
        return {
            "total_correct": stats["correct"],
            "total_incorrect": stats["incorrect"],
            "pending": stats["pending"],
            "accuracy": f"{stats['accuracy']:.1%}",
            "model_scores": scores,
        }