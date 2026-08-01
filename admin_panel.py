"""
admin_panel.py - 管理员审核面板
查看截图证据、判断正误、查看模型评分
"""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os, threading


class AdminPanel:
    """管理员审核面板 - 独立窗口"""

    def __init__(self, db, reinforcement, logger,
                 on_judge_callback=None):
        self._db = db
        self._rl = reinforcement
        self._logger = logger
        self._on_judge = on_judge_callback
        self._current_evidence = None
        self._window = None
        self._build()

    def _build(self):
        self._window = tk.Toplevel()
        self._window.title("管理员审核面板 - AI抽烟检测")
        self._window.geometry("900x680")
        self._window.configure(bg="#F5F8FC")

        # 顶部标题
        top = tk.Frame(self._window, bg="#1E5AA8", height=45)
        top.pack(fill=tk.X)
        tk.Label(top, text="管理员审核面板 - 截图证据判断",
                 fg="white", bg="#1E5AA8",
                 font=("Microsoft YaHei", 14, "bold")).pack(side=tk.LEFT, padx=15, pady=8)

        self._stats_label = tk.Label(top, text="", fg="#B0C4DE", bg="#1E5AA8",
                                     font=("Consolas", 10))
        self._stats_label.pack(side=tk.RIGHT, padx=15, pady=8)

        # 主体
        body = tk.Frame(self._window, bg="#F5F8FC")
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 左侧：截图预览
        left = tk.Frame(body, bg="white", width=500, relief=tk.RIDGE, bd=1)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        left.pack_propagate(False)
        tk.Label(left, text="截图证据", bg="white",
                 font=("Microsoft YaHei", 12, "bold"),
                 anchor="w").pack(fill=tk.X, padx=10, pady=(10, 5))
        self._image_label = tk.Label(left, bg="#1A1A1A", text="暂无待审核截图",
                                     font=("Microsoft YaHei", 12), fg="#888")
        self._image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 信息区
        self._info_text = tk.Text(left, height=6, bg="white",
                                  font=("Consolas", 10), state=tk.DISABLED)
        self._info_text.pack(fill=tk.X, padx=10, pady=5)

        # 右侧：操作面板
        right = tk.Frame(body, bg="white", width=280, relief=tk.RIDGE, bd=1)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        tk.Label(right, text="判断操作", bg="white",
                 font=("Microsoft YaHei", 12, "bold"),
                 anchor="w").pack(fill=tk.X, padx=10, pady=(10, 15))

        # 正确按钮
        self._btn_correct = tk.Button(right, text="✓ 判断正确 (+1分)",
                                       bg="#059669", fg="white",
                                       font=("Microsoft YaHei", 12, "bold"),
                                       command=self._judge_correct,
                                       state=tk.DISABLED, height=2)
        self._btn_correct.pack(fill=tk.X, padx=15, pady=5)

        # 错误按钮
        self._btn_wrong = tk.Button(right, text="✗ 判断错误 (-1分)",
                                     bg="#DC2626", fg="white",
                                     font=("Microsoft YaHei", 12, "bold"),
                                     command=self._judge_wrong,
                                     state=tk.DISABLED, height=2)
        self._btn_wrong.pack(fill=tk.X, padx=15, pady=5)

        # 分隔
        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=15, pady=10)

        # 备注
        tk.Label(right, text="备注:", bg="white",
                 font=("Microsoft YaHei", 10), anchor="w").pack(fill=tk.X, padx=15, pady=(0, 2))
        self._note_entry = tk.Text(right, height=3, font=("Microsoft YaHei", 10))
        self._note_entry.pack(fill=tk.X, padx=15, pady=5)

        # 模型评分
        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=15, pady=5)
        tk.Label(right, text="模型评分", bg="white",
                 font=("Microsoft YaHei", 12, "bold"),
                 anchor="w").pack(fill=tk.X, padx=10, pady=(5, 5))
        self._score_text = tk.Text(right, height=6, bg="white",
                                   font=("Consolas", 9), state=tk.DISABLED)
        self._score_text.pack(fill=tk.X, padx=10, pady=5)

        # 底部按钮
        bottom = tk.Frame(right, bg="white")
        bottom.pack(fill=tk.X, padx=15, pady=10)
        tk.Button(bottom, text="刷新列表", bg="#3B82F6", fg="white",
                  font=("Microsoft YaHei", 10),
                  command=self._refresh).pack(side=tk.LEFT, padx=2)
        tk.Button(bottom, text="跳过", bg="#6B7280", fg="white",
                  font=("Microsoft YaHei", 10),
                  command=self._skip).pack(side=tk.RIGHT, padx=2)

        self._refresh()

    def _refresh(self):
        """刷新待审核列表"""
        pending = self._db.get_pending_evidence()
        if pending:
            self._load_evidence(pending[0])
        else:
            self._current_evidence = None
            self._image_label.config(image="", text="暂无待审核截图\n\n所有截图已审核完毕!",
                                     font=("Microsoft YaHei", 14), fg="#059669")
            self._btn_correct.config(state=tk.DISABLED)
            self._btn_wrong.config(state=tk.DISABLED)
            self._info_text.config(state=tk.NORMAL)
            self._info_text.delete(1.0, tk.END)
            self._info_text.config(state=tk.DISABLED)

        self._update_scores()

    def _load_evidence(self, evidence):
        self._current_evidence = evidence

        # 加载图片
        img_path = evidence["image_path"]
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                img.thumbnail((480, 400))
                photo = ImageTk.PhotoImage(img)
                self._image_label.config(image=photo, text="")
                self._image_label.image = photo
            except Exception:
                self._image_label.config(image="", text="图片加载失败")

        # 显示信息
        info = f"摄像头: {evidence.get('camera_name', 'N/A')}\n"
        info += f"时间: {evidence.get('timestamp', 'N/A')}\n"
        info += f"检测类型: {evidence.get('label', 'N/A')}\n"
        info += f"置信度: {evidence.get('confidence', 0):.1%}\n"
        info += f"人物ID: {evidence.get('person_id', 'N/A')}"

        self._info_text.config(state=tk.NORMAL)
        self._info_text.delete(1.0, tk.END)
        self._info_text.insert(1.0, info)
        self._info_text.config(state=tk.DISABLED)

        self._btn_correct.config(state=tk.NORMAL)
        self._btn_wrong.config(state=tk.NORMAL)

    def _judge_correct(self):
        self._do_judge(True)

    def _judge_wrong(self):
        self._do_judge(False)

    def _do_judge(self, is_correct):
        if not self._current_evidence:
            return
        eid = self._current_evidence["id"]
        note = self._note_entry.get(1.0, tk.END).strip()
        self._db.judge_evidence(eid, is_correct, note)

        # 强化学习反馈
        label = self._current_evidence["label"]
        conf = self._current_evidence["confidence"]
        self._rl.process_feedback(label, is_correct, conf)

        result = "正确" if is_correct else "错误"
        self._logger.info(f"管理员判断: #{eid} -> {result} ({label})")
        self._note_entry.delete(1.0, tk.END)

        if self._on_judge:
            self._on_judge(eid, is_correct)

        self._refresh()

    def _skip(self):
        if not self._current_evidence:
            return
        pending = self._db.get_pending_evidence()
        if len(pending) > 1:
            # 移到下一个
            idx = next((i for i, e in enumerate(pending) if e["id"] == self._current_evidence["id"]), -1)
            next_idx = (idx + 1) % len(pending)
            self._load_evidence(pending[next_idx])
        else:
            self._refresh()

    def _update_scores(self):
        stats = self._db.get_stats_for_display()
        rl_stats = self._rl.get_stats()

        self._stats_label.config(
            text=f"正确:{stats['total_correct']} | 错误:{stats['total_incorrect']} | "
                 f"准确率:{stats['accuracy']} | 待审:{stats['pending']}"
        )

        score_text = f"模型评分:\n"
        for s in stats["model_scores"]:
            score_text += f"  {s['label']}: {s['accuracy']:.1%} "
            score_text += f"(+{s['total_correct']}/-{s['total_incorrect']})\n"

        score_text += f"\n自适应阈值:\n"
        for label, th in rl_stats["thresholds"].items():
            score_text += f"  {label}: {th:.3f}\n"
        score_text += f"学习轮次: {rl_stats['total_episodes']}\n"
        score_text += f"学习率: {rl_stats['learning_rate']}"

        self._score_text.config(state=tk.NORMAL)
        self._score_text.delete(1.0, tk.END)
        self._score_text.insert(1.0, score_text)
        self._score_text.config(state=tk.DISABLED)

    def show(self):
        if self._window:
            self._window.deiconify()
            self._refresh()

    def hide(self):
        if self._window:
            self._window.withdraw()