"""
build.py - 打包为独立应用 + 创建桌面快捷方式
"""
import os, sys, shutil, subprocess

APP_NAME = "视觉安防系统"
DEST_DIR = r"D:\视觉安防系统"
SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def build_exe():
    print("=" * 60)
    print(f"  {APP_NAME} - 打包构建")
    print("=" * 60)

    # 确保目标目录存在
    os.makedirs(DEST_DIR, exist_ok=True)

    # 复制核心文件
    files_to_copy = [
        "main.py", "headless_mode.py", "gui.py", "detector.py",
        "camera_manager.py", "decision_engine.py", "alerter.py",
        "logger.py", "database.py", "reinforcement.py",
        "person_identifier.py", "person_tracker.py",
        "camera_scanner.py", "admin_panel.py",
        "config.yaml", "requirements.txt", "train_offline.py",
        "generate_dataset.py", "model_downloader.py",
    ]

    print("复制源代码文件...")
    for f in files_to_copy:
        src = os.path.join(SRC_DIR, f)
        dst = os.path.join(DEST_DIR, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  {f}")

    # 复制模型文件
    models_dir = os.path.join(DEST_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    for m in ["smoking.onnx", "yolov8n.onnx", "rl_params.json"]:
        src = os.path.join(SRC_DIR, "models", m)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(models_dir, m))
            print(f"  models/{m}")

    # 创建数据目录
    os.makedirs(os.path.join(DEST_DIR, "alerts"), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, "logs"), exist_ok=True)

    # 创建启动批处理
    run_bat = os.path.join(DEST_DIR, "启动视觉安防系统.bat")
    with open(run_bat, "w", encoding="gbk") as f:
        f.write(f"""@echo off
chcp 65001 >nul
title {APP_NAME}
echo ============================================
echo   {APP_NAME}
echo   启动中...
echo ============================================
echo.
echo [1] 启动带界面的监控模式
echo [2] 启动后台无界面模式
echo [3] 打开管理员审核面板
echo [4] 退出
echo.
set /p choice="请选择 (1-4): "
if "%choice%"=="1" goto gui
if "%choice%"=="2" goto headless
if "%choice%"=="3" goto admin
if "%choice%"=="4" goto end
goto end

:gui
echo 启动带界面监控模式...
python main.py
goto end

:headless
echo 启动后台无界面模式...
echo 系统将在后台运行，截图保存到 alerts 目录
echo 按 Ctrl+C 停止
python headless_mode.py
goto end

:admin
echo 启动管理员审核面板...
python -c "from database import FeedbackDB; from reinforcement import ReinforcementLearner; from admin_panel import AdminPanel; import tkinter as tk; db=FeedbackDB('alerts/feedback.db'); rl=ReinforcementLearner(db); root=tk.Tk(); root.withdraw(); panel=AdminPanel(db, rl, None); root.mainloop()"
goto end

:end
echo 再见!
pause
""")
    print(f"  启动脚本: 启动视觉安防系统.bat")

    # 创建桌面快捷方式
    create_shortcut(run_bat)

    # 创建运行说明
    readme = os.path.join(DEST_DIR, "使用说明.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(f"""{'=' * 50}
  {APP_NAME} - 使用说明
{'=' * 50}

启动方式:
  1. 双击 "启动视觉安防系统.bat" 选择运行模式
  2. 或直接运行:
     - python main.py          (带界面)
     - python headless_mode.py (后台模式)

功能:
  - 多级检测: 人体追踪 -> ROI -> 香烟/烟盒检测
  - 强化学习: 管理员审核反馈自动调整检测阈值
  - 智能去重: 30秒内同一人物不重复保存截图
  - 后台运行: 无需GUI即可持续检测并保存截图
  - 多摄像头: 同时支持USB和RTSP网络摄像头

目录结构:
  alerts/ - 检测截图存证
  logs/   - 运行日志
  models/ - AI模型文件

配置:
  编辑 config.yaml 修改摄像头、检测参数等

管理员审核:
  双击启动脚本选择 [3] 打开审核面板
  或运行: python admin_panel_standalone.py

{'=' * 50}
""")
    print(f"  使用说明: 使用说明.txt")

    print(f"\n打包完成! 位置: {DEST_DIR}")
    print(f"桌面快捷方式: {APP_NAME}.lnk")
    print("=" * 60)


def create_shortcut(target_path):
    """创建桌面快捷方式"""
    try:
        import pythoncom
        from win32com.client import Dispatch

        pythoncom.CoInitialize()
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        shortcut_path = os.path.join(desktop, f"{APP_NAME}.lnk")

        shell = Dispatch("WScript.Shell")
        shortcut = shell.CreateShortcut(shortcut_path)
        shortcut.TargetPath = target_path
        shortcut.WorkingDirectory = DEST_DIR
        shortcut.Description = f"{APP_NAME} - AI抽烟检测智能体"
        shortcut.IconLocation = "shell32.dll,21"
        shortcut.Save()

        print(f"  桌面快捷方式已创建: {shortcut_path}")
    except ImportError:
        print("  警告: 未安装 pywin32，无法创建快捷方式")
        print(f"  请手动创建快捷方式指向: {target_path}")
    except Exception as e:
        print(f"  快捷方式创建失败: {e}")
        print(f"  请手动创建快捷方式指向: {target_path}")


def run_pyinstaller():
    """使用 PyInstaller 打包为 exe"""
    print("\n使用 PyInstaller 打包为独立 exe...")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile", "--noconsole",
        "--name", "视觉安防系统",
        "--add-data", f"config.yaml;.",
        "--add-data", f"models;models",
        "--hidden-import", "onnxruntime",
        "--hidden-import", "cv2",
        "--hidden-import", "PIL",
        "--hidden-import", "yaml",
        "main.py",
    ]
    subprocess.run(cmd, cwd=SRC_DIR)
    print("PyInstaller 打包完成!")


if __name__ == "__main__":
    build_exe()
    # 如需打包为独立 exe（需要安装 PyInstaller），取消下行注释:
    # run_pyinstaller()