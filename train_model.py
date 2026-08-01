"""
train_model.py - 自动下载抽烟检测数据集并训练 ONNX 模型
完全自动化流程：下载数据集 → 训练 YOLO → 导出 ONNX

数据来源:
  - Roboflow Universe 公开数据集 (需要 API key: https://roboflow.com 免费注册)
  - 或使用已有本地数据集

用法:
  python train_model.py --api-key YOUR_ROBOFLOW_API_KEY
  python train_model.py --dataset ./your_dataset
"""
import os
import sys
import argparse
import shutil


def train_with_roboflow(api_key: str):
    """从 Roboflow 下载公开抽烟检测数据集并训练"""
    print("=" * 60)
    print("  步骤 1/3: 从 Roboflow 下载抽烟检测数据集")
    print("=" * 60)

    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)

    # 多个公开的抽烟检测数据集，按优先级尝试
    datasets = [
        # Smoking Detection Dataset (多类别: cigarette, person, smoke, vape)
        {"ws": "smoking-detection-kxrvo", "proj": "cigarette-detection-0mote", "ver": 3},
        # 备用
        {"ws": "project-7vb7f", "proj": "smoking-detection-2bm4l", "ver": 1},
    ]

    dataset_path = None
    for ds in datasets:
        try:
            print(f"  尝试下载: {ds['ws']}/{ds['proj']} v{ds['ver']}...")
            project = rf.workspace(ds["ws"]).project(ds["proj"])
            dataset = project.version(ds["ver"]).download("yolov8", location="./datasets")
            dataset_path = dataset.location
            print(f"  下载成功: {dataset_path}")
            break
        except Exception as e:
            print(f"  下载失败: {e}")
            continue

    if dataset_path is None:
        print("\n  无法自动下载数据集，请手动下载:")
        print("  1. 访问 https://universe.roboflow.com/")
        print("  2. 搜索 'smoking detection' 或 'cigarette detection'")
        print("  3. 下载 YOLOv8 格式并解压到 ./datasets/")
        print("  4. 运行: python train_model.py --dataset ./datasets/数据集目录")
        return False

    return _train_and_export(dataset_path)


def train_with_local(dataset_path: str):
    """使用本地数据集训练"""
    if not os.path.exists(dataset_path):
        print(f"  数据集路径不存在: {dataset_path}")
        return False

    data_yaml = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"  data.yaml 不存在，尝试查找...")
        # 可能嵌套了一层
        for root, dirs, files in os.walk(dataset_path):
            if "data.yaml" in files:
                data_yaml = os.path.join(root, "data.yaml")
                dataset_path = root
                break

    if not os.path.exists(data_yaml):
        print(f"  找不到 data.yaml，请确认数据集路径正确")
        return False

    print(f"  数据集: {dataset_path}")
    return _train_and_export(dataset_path)


def _train_and_export(dataset_path: str) -> bool:
    """训练模型并导出 ONNX"""
    from ultralytics import YOLO

    # 查找 data.yaml
    data_yaml = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml):
        for root, dirs, files in os.walk(dataset_path):
            if "data.yaml" in files:
                data_yaml = os.path.join(root, "data.yaml")
                break

    print(f"\n{'=' * 60}")
    print("  步骤 2/3: 训练 YOLOv8n 抽烟检测模型")
    print(f"{'=' * 60}")

    print("  加载 YOLOv8n 预训练权重...")
    model = YOLO("yolov8n.pt")

    print("  开始训练 (epochs=30, imgsz=416, batch=8)...")
    print("  提示: 训练时间取决于数据集大小和硬件，请耐心等待...")
    print()

    results = model.train(
        data=data_yaml,
        epochs=30,
        imgsz=416,
        batch=8,
        name="smoking_det",
        project="runs",
        exist_ok=True,
        workers=0,
        verbose=True,
    )

    print(f"\n{'=' * 60}")
    print("  步骤 3/3: 导出 ONNX 模型")
    print(f"{'=' * 60}")

    best_pt = "runs/smoking_det/weights/best.pt"
    if not os.path.exists(best_pt):
        print("  训练权重未找到，请检查训练日志")
        return False

    print("  导出 ONNX (FP16, 416x416)...")
    model = YOLO(best_pt)
    model.export(format="onnx", imgsz=416, half=True)

    # 移动到 models/ 目录
    os.makedirs("models", exist_ok=True)
    src = "runs/smoking_det/weights/best.onnx"
    if os.path.exists(src):
        shutil.copy(src, "models/smoking.onnx")
        print(f"  模型已导出: models/smoking.onnx")
        print(f"  文件大小: {os.path.getsize('models/smoking.onnx') / 1024 / 1024:.1f} MB")
    else:
        print("  ONNX 导出失败")
        return False

    print(f"\n{'=' * 60}")
    print("  训练完成！")
    print(f"  模型文件: models/smoking.onnx")
    print(f"  现在可以运行: python main.py")
    print(f"{'=' * 60}")
    return True


def main():
    parser = argparse.ArgumentParser(description="抽烟检测模型自动训练工具")
    parser.add_argument("--api-key", type=str, help="Roboflow API Key (免费注册: https://roboflow.com)")
    parser.add_argument("--dataset", type=str, help="本地数据集路径")
    args = parser.parse_args()

    if args.dataset:
        success = train_with_local(args.dataset)
    elif args.api_key:
        success = train_with_roboflow(args.api_key)
    else:
        print("""
  请提供数据来源:

  方式一: 从 Roboflow 下载 (推荐)
  ─────────────────────────────
  1. 访问 https://roboflow.com 免费注册
  2. 获取 API Key
  3. 运行: python train_model.py --api-key YOUR_KEY

  方式二: 使用本地数据集
  ─────────────────────────
  python train_model.py --dataset ./datasets/你的数据集

  方式三: 手动下载数据集
  ─────────────────────────
  1. 访问 https://universe.roboflow.com/
  2. 搜索 "smoking detection"
  3. 下载 YOLOv8 格式，解压到 ./datasets/
  4. 运行: python train_model.py --dataset ./datasets/数据集目录
  """)
        sys.exit(1)

    if not success:
        print("\n  训练失败，请检查以上错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()