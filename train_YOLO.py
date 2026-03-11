from ultralytics import YOLO
import os
from datetime import datetime


def train_yolo(
        # 基础配置（核心必改参数）
        model_path: str = "H:\lab\yolov5-master\yolov8n.pt",  # 预训练模型路径（yolov8n/v/m/l/x.pt 或自定义权重）
        data_config: str = "expression\loopy.yaml",  # 数据集配置文件路径（关键！需自行准备）
        output_dir: str = None,  # 自定义输出目录（None 时自动生成）
        fraction: float = 1.0,
        workers=4,
        resume: bool = False,
        # 训练参数（按需修改）
        epochs: int = 100,  # 训练轮数
        batch: int = 16,  # 批次大小（根据GPU显存调整）
        imgsz: int = 640,  # 输入图像尺寸
        device: str = "0",  # 训练设备（"0"=单GPU, "0,1"=多GPU, "cpu"=CPU）
        lr0: float = 0.01,  # 初始学习率
        lrf: float = 0.01,  # 最终学习率因子（lr0 * lrf）
        weight_decay: float = 0.0005,  # 权重衰减（防止过拟合）
        momentum: float = 0.937,  # 动量
        patience: int = 50,  # 早停耐心值（多少轮无提升停止）
        save: bool = True,  # 是否保存检查点
        save_period: int = -1,  # 保存检查点周期（-1=只保存最佳）
        val: bool = True,  # 训练时是否验证
        plots: bool = True,  # 是否生成训练可视化图表
        augment: bool = True,  # 是否使用默认数据增强
        mixup: float = 0.0,  # mixup概率（0-1，增强泛化性）
        mosaic: float = 1.0,  # mosaic增强概率（0-1）
        conf: float = 0.001,  # 检测置信度阈值
        iou: float = 0.6,  # NMS的IOU阈值
        classes: list = None,  # 只训练/检测特定类别（例：[0,2] 只训练第0、2类）
        single_cls: bool = False,  # 是否单类别训练
        rect: bool = False,  # 是否使用矩形训练（加速）
        cos_lr: bool = False,  # 是否使用余弦学习率调度器
        pretrained: bool = True,  # 是否使用预训练权重（False=从头训练）
        optimizer: str = "SGD",  # 优化器（SGD, Adam, AdamW, RMSProp）
        warmup_epochs: float = 3.0,  # 热身轮数
        warmup_momentum: float = 0.8,  # 热身动量
        warmup_bias_lr: float = 0.1,  # 热身偏置学习率
        box: float = 7.5,  # 边界框损失权重
        cls: float = 0.5,  # 分类损失权重
        dfl: float = 1.5,  # 分布焦点损失权重
        hsv_h: float = 0.015,  # HSV色调增强幅度
        hsv_s: float = 0.7,  # HSV饱和度增强幅度
        hsv_v: float = 0.4,  # HSV明度增强幅度
        degrees: float = 0.0,  # 旋转角度（-degrees 到 degrees）
        translate: float = 0.1,  # 平移幅度（占图像宽高的比例）
        scale: float = 0.5,  # 缩放幅度（0.5=缩小50%到放大100%）
        shear: float = 0.0,  # 剪切角度
        perspective: float = 0.0,  # 透视变换幅度
        flipud: float = 0.0,  # 上下翻转概率
        fliplr: float = 0.5,  # 左右翻转概率
        copy_paste: float = 0.0,  # Copy-paste增强概率
) -> None:
    """
    通用YOLO训练函数（基于Ultralytics）
    支持自定义输出目录、灵活调整训练参数
    """
    # ---------------------- 初始化输出目录 ----------------------
    if output_dir is None:
        # 自动生成输出目录（格式：runs/train/自定义前缀_时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("runs", "train", f"exp_{timestamp}")
    else:
        # 自定义输出目录（若不存在则创建）
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    print(f"📁 训练输出将保存到：{output_dir}")

    # ---------------------- 加载模型 ----------------------
    print(f"🔧 加载模型：{model_path}")
    model = YOLO(model_path)

    # ---------------------- 配置训练参数 ----------------------
    train_args = {
        "data": data_config,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "lr0": lr0,
        "lrf": lrf,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "patience": patience,
        "save": save,
        "save_period": save_period,
        "val": val,
        "plots": plots,
        "augment": augment,
        "mixup": mixup,
        "mosaic": mosaic,
        "conf": conf,
        "iou": iou,
        "classes": classes,
        "single_cls": single_cls,
        "rect": rect,
        "cos_lr": cos_lr,
        "pretrained": pretrained,
        "optimizer": optimizer,
        "warmup_epochs": warmup_epochs,
        "warmup_momentum": warmup_momentum,
        "warmup_bias_lr": warmup_bias_lr,
        "box": box,
        "cls": cls,
        "dfl": dfl,
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
        "degrees": degrees,
        "translate": translate,
        "scale": scale,
        "shear": shear,
        "perspective": perspective,
        "flipud": flipud,
        "fliplr": fliplr,
        "mosaic": mosaic,
        "mixup": mixup,
        "copy_paste": copy_paste,
        "project": os.path.dirname(output_dir),  # 输出根目录
        "name": os.path.basename(output_dir),  # 输出子目录名称
        "exist_ok": True,  # 允许覆盖已有目录
        "fraction": fraction,
        "workers": workers,
        "resume": resume,
    }

    # ---------------------- 开始训练 ----------------------
    print("🚀 开始YOLO训练...")
    print(f"📋 训练参数摘要：epochs={epochs}, batch={batch}, imgsz={imgsz}, device={device}")
    results = model.train(**train_args)

    # ---------------------- 训练完成后输出信息 ----------------------
    print("🎉 训练完成！")
    print(f"📊 训练结果文件位置：{output_dir}")
    print(f"🏆 最佳权重文件：{os.path.join(output_dir, 'best.pt')}")
    print(f"📈 训练日志：{os.path.join(output_dir, 'results.csv')}")


if __name__ == "__main__":
    # ---------------------- 这里是参数修改入口（根据需求调整） ----------------------
    train_yolo(
        # 基础配置（必改）
        model_path="yolov9s.pt",  # 可改为 yolov8s/v/m/l/x.pt（模型越大效果越好但速度越慢）
        data_config="expression/loopy.yaml",  # 改为你的数据集配置文件路径（关键！）
        output_dir="YOLOv9s_training_results",  # 自定义输出目录（改为你想要的路径）
        fraction=0.3,
        workers=4,
        #resume=True,  # 关键：启用恢复训练
        # 训练核心参数（常用修改）
        epochs=100,  # 训练轮数（新手建议50-100，复杂数据集可增加到200）
        batch=2,  # 批次大小（GPU显存不足就改小，如4；显存充足改大，如32）
        imgsz=320,  # 输入图像尺寸（常用640/800/1024，需是32的倍数）
        device="0",  # 训练设备（"0"=GPU, "cpu"=CPU，多GPU写"0,1"）
        pretrained=True,  # 是否使用预训练权重（False=从头训练，速度慢效果差）
        # 优化器和学习率（按需调整）
        optimizer="AdamW",  # 优化器（AdamW在小数据集上可能比SGD效果好）
        lr0=0.001,  # 初始学习率（AdamW建议0.001，SGD建议0.01）

        # 数据增强（防止过拟合）
        #augment=True,  # 启用默认增强
        mixup=0.2,  # 启用mixup增强（0.2=20%概率，增加泛化性）
        fliplr=0.5,  # 左右翻转概率（0.5=50%）

        # 损失权重（复杂数据集可调整）
        box=7.5,  # 边界框损失权重（越高越注重框的准确性）
        cls=1.0,  # 分类损失权重（越高越注重类别的准确性）
    )