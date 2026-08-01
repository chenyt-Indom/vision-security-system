"""
gui.py - 桌面 GUI 界面（重写版）
修复视频显示 + 管理员面板 + 强化学习 + 智能去重
"""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2, threading, time, os, numpy as np
from person_identifier import PersonIdentifier


class DetectionGUI:
    def __init__(self, config, camera_manager, detector, decision_engine,
                 alerter, logger, db, reinforcement):
        self._cfg = config["ui"]
        self._perf_cfg = config.get("performance", {})
        self._model_cfg = config.get("model", {})
        self._cameras = camera_manager
        self._detector = detector
        self._decision = decision_engine
        self._alerter = alerter
        self._logger = logger
        self._db = db
        self._rl = reinforcement
        self._running = False
        self._selected_cam = None
        self._alert_records = []
        self._frame_count = 0
        self._person_id = PersonIdentifier(time_window=30)
        self._photo_ref = None  # 防止 GC 回收
        self._build_window()

    def _build_window(self):
        self._root = tk.Tk()
        self._root.title(self._cfg["title"])
        w, h = self._cfg["window_size"]
        self._root.geometry(f"{w}x{h}")
        self._root.configure(bg="#F5F8FC")

        # ===== 顶部标题栏 =====
        topbar = tk.Frame(self._root, bg="#1E5AA8", height=50)
        topbar.pack(fill=tk.X)
        tk.Label(topbar, text="AI 抽烟检测系统 - 安防智能体",
                 fg="white", bg="#1E5AA8",
                 font=("Microsoft YaHei", 16, "bold")).pack(side=tk.LEFT, padx=15, pady=10)

        self._perf_label = tk.Label(topbar, text="", fg="#B0C4DE", bg="#1E5AA8",
                                    font=("Consolas", 9))
        self._perf_label.pack(side=tk.LEFT, padx=10)

        # 管理员面板按钮
        tk.Button(topbar, text="审核面板", bg="#F59E0B", fg="white",
                  font=("Microsoft YaHei", 10),
                  command=self._open_admin).pack(side=tk.RIGHT, padx=5, pady=10)

        self._btn_start = tk.Button(topbar, text="▶ 开始检测", bg="#059669", fg="white",
                                    font=("Microsoft YaHei", 11), command=self.start)
        self._btn_start.pack(side=tk.RIGHT, padx=(5, 5), pady=10)

        self._btn_stop = tk.Button(topbar, text="■ 停止检测", bg="#DC2626", fg="white",
                                   font=("Microsoft YaHei", 11), command=self.stop,
                                   state=tk.DISABLED)
        self._btn_stop.pack(side=tk.RIGHT, padx=5, pady=10)

        # ===== 主体 =====
        body = tk.Frame(self._root, bg="#F5F8FC")
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 左侧摄像头列表
        left = tk.Frame(body, bg="white", width=200, relief=tk.RIDGE, bd=1)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left.pack_propagate(False)
        tk.Label(left, text="摄像头列表", bg="white",
                 font=("Microsoft YaHei", 12, "bold"),
                 anchor="w").pack(fill=tk.X, padx=10, pady=(10, 5))
        self._cam_list_frame = tk.Frame(left, bg="white")
        self._cam_list_frame.pack(fill=tk.BOTH, expand=True, padx=5)

        # 右侧视频画面
        right = tk.Frame(body, bg="#1A1A1A")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self._video_label = tk.Label(right, bg="#1A1A1A")
        self._video_label.pack(fill=tk.BOTH, expand=True)

        # 底部告警记录
        bottom = tk.Frame(self._root, bg="white", height=120, relief=tk.RIDGE, bd=1)
        bottom.pack(fill=tk.X, padx=10, pady=(0, 10))
        bottom.pack_propagate(False)
        tk.Label(bottom, text="告警记录", bg="white",
                 font=("Microsoft YaHei", 12, "bold"),
                 anchor="w").pack(fill=tk.X, padx=10, pady=(5, 0))
        self._alert_list = tk.Text(bottom, height=4, bg="white",
                                   font=("Consolas", 10),
                                   state=tk.DISABLED, wrap=tk.WORD)
        self._alert_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 管理员面板引用
        self._admin_panel = None
        from admin_panel import AdminPanel
        self._admin_panel = AdminPanel(self._db, self._rl, self._logger,
                                       on_judge_callback=self._on_admin_judge)

    def _open_admin(self):
        if self._admin_panel:
            self._admin_panel.show()

    def _on_admin_judge(self, evidence_id, is_correct):
        """管理员判断后更新阈值"""
        new_thresholds = self._rl.get_stats()["thresholds"]
        self._logger.info(f"RL阈值更新: {new_thresholds}")

    def _refresh_camera_list(self):
        for w in self._cam_list_frame.winfo_children():
            w.destroy()
        for cam in self._cameras.all_cameras:
            color = "#059669" if cam.enabled else "#9CA3AF"
            status = "●" if cam.enabled else "○"
            text = f"{status} {cam.name}"
            btn = tk.Button(self._cam_list_frame, text=text, bg="white", fg=color,
                            font=("Microsoft YaHei", 10), anchor="w", relief=tk.FLAT,
                            command=lambda c=cam: self._select_camera(c))
            btn.pack(fill=tk.X, pady=3, padx=5)

    def _select_camera(self, cam):
        self._selected_cam = cam.id

    def start(self):
        self._running = True
        self._btn_start.config(state=tk.DISABLED)
        self._btn_stop.config(state=tk.NORMAL)
        self._refresh_camera_list()
        self._logger.info("检测已启动")
        threading.Thread(target=self._loop, daemon=True).start()
        self._update_perf()

    def stop(self):
        self._running = False
        self._btn_start.config(state=tk.NORMAL)
        self._btn_stop.config(state=tk.DISABLED)
        self._logger.info("检测已停止")

    def _update_perf(self):
        if not self._running:
            return
        backend = self._detector.backend_name
        inf_ms = self._detector.avg_inference_time_ms
        stats = self._db.get_stats_for_display()
        self._perf_label.config(
            text=f"后端:{backend} | 推理:{inf_ms:.1f}ms | "
                 f"准确率:{stats['accuracy']} | 待审:{stats['pending']}"
        )
        self._root.after(1000, self._update_perf)

    def _loop(self):
        """检测主循环"""
        for cam in self._cameras.active_cameras:
            try:
                cam.start()
            except RuntimeError as e:
                self._logger.info(str(e))

        active = self._cameras.active_cameras
        if active and not self._selected_cam:
            self._selected_cam = active[0].id

        interval = self._perf_cfg.get("inference_interval", 1)
        frame_idx = 0
        last_alert_time = {}  # 去重：每个摄像头+人物ID的最后告警时间

        while self._running:
            for cam in self._cameras.active_cameras:
                frame = cam.read()
                if frame is None:
                    continue

                frame_idx += 1
                display_frame = frame.copy()

                if frame_idx % interval == 0:
                    # === 多级检测 ===
                    result = self._detector.detect(frame)

                    # S1: 人体框 (绿色)
                    for p in result["persons"]:
                        x1, y1, x2, y2 = p["bbox"]
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 200, 0), 3)
                        cv2.putText(display_frame, f"Person#{p['id']}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

                    # S2: ROI 区域 (蓝色虚线)
                    for roi in result["rois"]:
                        rx1, ry1, rx2, ry2 = roi["bbox"]
                        for dx in range(rx1, rx2, 12):
                            cv2.line(display_frame, (dx, ry1), (min(dx + 6, rx2), ry1),
                                     (200, 150, 50), 1)
                            cv2.line(display_frame, (dx, ry2), (min(dx + 6, rx2), ry2),
                                     (200, 150, 50), 1)
                        for dy in range(ry1, ry2, 12):
                            cv2.line(display_frame, (rx1, dy), (rx1, min(dy + 6, ry2)),
                                     (200, 150, 50), 1)
                            cv2.line(display_frame, (rx2, dy), (rx2, min(dy + 6, ry2)),
                                     (200, 150, 50), 1)
                        cv2.putText(display_frame, f"{roi['type']} ROI", (rx1, ry1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 150, 50), 1)

                    # S3: 告警框 (红色/橙色) + 智能去重
                    alert_detected = False
                    for alert in result["alerts"]:
                        ax1, ay1, ax2, ay2 = alert["bbox"]
                        label = alert["label"]
                        conf = alert["confidence"]
                        tid = alert["track_id"]

                        # 自适应阈值
                        rl_threshold = self._rl.get_threshold(label)
                        if conf < rl_threshold:
                            continue

                        # 智能去重：同一人物短时间内只保留最高置信度
                        person_key = f"{cam.id}_{tid}"
                        now = time.time()

                        if person_key in last_alert_time:
                            last_time, last_conf, last_bbox = last_alert_time[person_key]
                            if now - last_time < 30:  # 30秒内
                                if conf > last_conf:
                                    # 更新为更高置信度
                                    last_alert_time[person_key] = (now, conf, alert["bbox"])
                                continue

                        last_alert_time[person_key] = (now, conf, alert["bbox"])

                        # 颜色
                        if "pack" in label:
                            color = (0, 165, 255)  # 橙色
                        else:
                            color = (0, 0, 255)    # 红色

                        cv2.rectangle(display_frame, (ax1, ay1), (ax2, ay2), color, 4)
                        text = f"ALERT: {label} {conf:.0%}"
                        cv2.rectangle(display_frame, (ax1, ay1 - 28), (ax2, ay1), color, -1)
                        cv2.putText(display_frame, text, (ax1 + 4, ay1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                        alert_detected = True

                    # === 决策链 + 证据保存 ===
                    should_alert, detail = self._decision.evaluate(cam.id, result["alerts"])
                    if should_alert and alert_detected:
                        best_alert = max(result["alerts"], key=lambda a: a["confidence"])
                        detail["detected_label"] = best_alert["label"]
                        detail["confidence"] = best_alert["confidence"]

                        # 提取人物特征用于去重
                        person_features = {}
                        track_id = best_alert.get("track_id", 0)
                        for p in result["persons"]:
                            if p["id"] == track_id:
                                person_features = self._person_id.extract_features(
                                    frame, p["bbox"])
                                break

                        # 保存截图证据
                        alert_dir = "alerts"
                        os.makedirs(alert_dir, exist_ok=True)
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        img_path = os.path.join(
                            alert_dir,
                            f"{cam.id}_{best_alert['label']}_{ts}.jpg")
                        cv2.imwrite(img_path, display_frame)

                        # 存入数据库
                        self._db.add_evidence(
                            cam.id, cam.name,
                            f"person_{track_id}",
                            img_path, best_alert["label"],
                            best_alert["confidence"],
                            best_alert["bbox"],
                            person_features,
                        )

                        # 更新告警列表
                        self._root.after(0, lambda c=cam, d=detail: self._on_alert(c, d))

                # 显示 FPS
                if self._perf_cfg.get("display_fps", True):
                    inf_ms = self._detector.avg_inference_time_ms
                    cv2.putText(display_frame, f"FPS: {self._detector.current_fps:.0f} | {inf_ms:.1f}ms",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # === 更新 GUI 画面（修复版） ===
                if cam.id == self._selected_cam:
                    try:
                        img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (860, 600))
                        photo = ImageTk.PhotoImage(Image.fromarray(img))
                        self._photo_ref = photo  # 保持引用防止 GC
                        self._root.after(0, self._safe_update_video, photo)
                    except Exception:
                        pass

                self._frame_count += 1

    def _safe_update_video(self, photo):
        """安全更新视频帧"""
        try:
            self._video_label.config(image=photo)
            self._video_label.image = photo
        except Exception:
            pass

    def _on_alert(self, cam, detail):
        max_history = self._cfg["alert_history_max"]
        label = detail.get("detected_label", "检测")
        conf = detail.get("confidence", 0)
        record = f"[{detail['timestamp']}] {cam.name}  {label}  置信度{conf:.0%}"
        self._alert_records.append(record)
        if len(self._alert_records) > max_history:
            self._alert_records = self._alert_records[-max_history:]

        self._alert_list.config(state=tk.NORMAL)
        self._alert_list.delete(1.0, tk.END)
        self._alert_list.insert(tk.END, "\n".join(self._alert_records))
        self._alert_list.see(tk.END)
        self._alert_list.config(state=tk.DISABLED)

    def run(self):
        self._refresh_camera_list()
        self._root.mainloop()