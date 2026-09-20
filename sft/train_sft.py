# -*- coding: utf-8 -*-
"""SFT 后训练脚本：加载预训练权重，在「指令 → 故事」数据上微调（只在故事部分计算损失）。

不修改任何预训练产物：权重从 out/model.pt 读取，微调结果保存到 out/model_sft.pt。

用法（在项目根目录 mimi_gpt/ 下执行）：
  python sft/train_sft.py                          # 默认配置（GPU 约 1000 iter，几分钟）
  python sft/train_sft.py --max_iters 300         # 快速验证用
  python sft/train_sft.py --model out/model_sft.pt  # 在已有 SFT 模型上继续训练
"""

import argparse  # 命令行参数解析库
import math  # 数学库，用于余弦学习率调度
import os  # 系统接口，用于拼接路径、创建目录
import sys  # 系统路径操作，用于把项目根加入模块搜索路径
import time  # 时间库，用于打印训练速度

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 必须在 import torch 之前设置：固定 cuBLAS 工作区（与 train.py 相同的稳定化措施）

import numpy as np  # 读取 int16 样本二进制文件
import torch  # PyTorch 核心库

torch.use_deterministic_algorithms(True, warn_only=True)  # 优先选用确定性 CUDA 内核（与 train.py 相同的稳定化措施）

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录 mimi_gpt/（本脚本位于 sft/ 子目录）
sys.path.insert(0, ROOT_DIR)  # 把项目根加入模块搜索路径，才能导入根目录的 model.py
from model import GPT, GPTConfig  # 导入模型定义

DATA_DIR = os.path.join(ROOT_DIR, "data")  # 数据目录（与预训练阶段共享）
OUT_DIR = os.path.join(ROOT_DIR, "out")  # checkpoint 输出目录（预训练 model.pt 与 SFT model_sft.pt 都在这里）

SFT_CONFIG = dict(batch_size=64, learning_rate=1e-4, min_lr=1e-5,  # 微调用小得多的学习率（预训练的 1/6），避免灾难性遗忘
                  warmup_iters=50, max_iters=1000, eval_interval=200, eval_iters=50)  # 微调轮次远小于预训练


def get_lr(it, cfg):  # 第 it 步的学习率：线性预热 + 余弦退火（与 train.py 同一套调度）
    if it < cfg["warmup_iters"]:  # 预热阶段
        return cfg["learning_rate"] * (it + 1) / cfg["warmup_iters"]  # 线性上升到目标学习率
    if it >= cfg["max_iters"]:  # 训练结束之后
        return cfg["min_lr"]  # 停在最低学习率
    ratio = (it - cfg["warmup_iters"]) / (cfg["max_iters"] - cfg["warmup_iters"])  # 预热完成后的进度 0..1
    return cfg["min_lr"] + 0.5 * (cfg["learning_rate"] - cfg["min_lr"]) * (1 + math.cos(math.pi * ratio))  # 余弦下降


def get_batch(ids_data, label_data, batch_size, device):  # 随机取一批 SFT 样本
    """返回 (x, y)：x 是补齐后的完整序列，y 是带 -100 掩码的标签（提示位置不计算损失）。"""
    n = ids_data.shape[0]  # 样本总数
    ix = torch.randint(n, (batch_size,))  # 随机抽取样本行号（每行已定长，无需再切窗口）
    x = torch.stack([torch.from_numpy(ids_data[i].astype(np.int64)) for i in ix])  # 输入序列
    y = torch.stack([torch.from_numpy(label_data[i].astype(np.int64)) for i in ix])  # 掩码标签
    return x.to(device), y.to(device)  # 搬到计算设备


@torch.no_grad()  # 评估阶段不需要梯度
def estimate_loss(model, ids_dict, label_dict, cfg, device, autocast_ctx):  # 在训练/验证集上估计平均损失
    """返回 {"train": ..., "val": ...} 的平均损失字典（只统计故事位置，-100 被自动忽略）。"""
    model.eval()  # 评估模式（关闭 dropout）
    out = {}  # 结果容器
    for split in ids_dict:  # 遍历训练/验证
        losses = torch.zeros(cfg["eval_iters"])  # 各批损失
        for k in range(cfg["eval_iters"]):  # 多批平均降低波动
            X, Y = get_batch(ids_dict[split], label_dict[split], cfg["batch_size"], device)  # 取一批
            with autocast_ctx:  # 与训练同精度
                _, loss = model(X, Y)  # 前向（模型内部交叉熵默认忽略 -100 位置）
            losses[k] = loss.item()  # 记录
        out[split] = losses.mean().item()  # 平均
    model.train()  # 切回训练模式
    return out  # 返回结果


def main():  # 主流程：加载预训练权重 → 读 SFT 数据 → 微调 → 保存
    parser = argparse.ArgumentParser(description="SFT 后训练 mimi_gpt")  # 命令行入口
    parser.add_argument("--model", type=str, default=os.path.join(OUT_DIR, "model.pt"), help="预训练 checkpoint 路径（只读，不会被修改）")  # 初始权重
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")  # 默认自动选
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖默认 batch_size")  # 兼容性/显存调节
    parser.add_argument("--max_iters", type=int, default=None, help="覆盖默认训练步数")  # 快速验证用
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp32"], help="训练精度：auto 在 GPU 上用 bf16 混合精度")  # 与 train.py 相同策略
    args = parser.parse_args()  # 解析参数

    cfg = dict(SFT_CONFIG)  # 复制一份配置（允许被命令行覆盖）
    if args.batch_size is not None:  # 指定了 batch_size 就覆盖
        cfg["batch_size"] = args.batch_size  # 应用覆盖值
    if args.max_iters is not None:  # 指定了步数就覆盖
        cfg["max_iters"] = args.max_iters  # 应用覆盖值

    device = args.device  # 计算设备
    use_bf16 = args.dtype == "bf16" or (args.dtype == "auto" and device.startswith("cuda"))  # GPU 默认 bf16 混合精度
    torch.manual_seed(1337)  # 固定随机种子

    ckpt = torch.load(args.model, map_location="cpu")  # 读取预训练 checkpoint（含配置与权重）
    model = GPT(GPTConfig(**ckpt["config"])).to(device)  # 按保存的配置重建模型并搬到设备
    model.load_state_dict(ckpt["model"])  # 载入预训练权重
    block_size = ckpt["config"]["block_size"]  # 上下文长度（数据打包时已按它定长）
    print(f"已加载预训练模型 {args.model}（val loss {ckpt['val_loss']:.4f}）| 精度 {'bf16 混合' if use_bf16 else 'fp32'}")  # 打印起点信息

    train_ids = np.memmap(os.path.join(DATA_DIR, "sft_train_ids.bin"), dtype=np.int16, mode="r").reshape(-1, block_size)  # 训练输入
    train_labels = np.memmap(os.path.join(DATA_DIR, "sft_train_labels.bin"), dtype=np.int16, mode="r").reshape(-1, block_size)  # 训练标签
    val_ids = np.memmap(os.path.join(DATA_DIR, "sft_val_ids.bin"), dtype=np.int16, mode="r").reshape(-1, block_size)  # 验证输入
    val_labels = np.memmap(os.path.join(DATA_DIR, "sft_val_labels.bin"), dtype=np.int16, mode="r").reshape(-1, block_size)  # 验证标签
    print(f"SFT 训练集 {train_ids.shape[0]:,} 条 | 验证集 {val_ids.shape[0]:,} 条")  # 打印数据规模

    decay, no_decay = [], []  # 参数分组：2D 权重做 decay，LayerNorm/偏置不做（与 train.py 相同）
    for p in model.parameters():  # 遍历参数
        (decay if p.dim() >= 2 else no_decay).append(p)  # 按 ndim 分组
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],  # 分组 weight decay
                                  lr=cfg["learning_rate"], betas=(0.9, 0.95))  # AdamW 优化器
    ids_dict = {"train": train_ids, "val": val_ids}  # 打包给评估函数
    label_dict = {"train": train_labels, "val": val_labels}  # 同上
    autocast_ctx = torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=use_bf16)  # 混合精度上下文
    model.train()  # 训练模式
    t0 = time.time()  # 计时起点
    for it in range(1, cfg["max_iters"] + 1):  # 主训练循环
        lr = get_lr(it, cfg)  # 当前学习率
        for g in optimizer.param_groups:  # 应用到所有参数组
            g["lr"] = lr  # 设置本步学习率
        x, y = get_batch(train_ids, train_labels, cfg["batch_size"], device)  # 取一批样本
        with autocast_ctx:  # 前向混合精度
            _, loss = model(x, y)  # 计算损失（提示位置被 -100 掩掉，只学习故事部分）
        if not torch.isfinite(loss):  # 出现 NaN/Inf
            print(f"iter {it}: loss = {loss.item()}，提前停止（不保存本次权重）")  # 提示
            if it == 1:  # 第一步就异常时用 CPU 复算对照，锁定问题来源
                cpu_model = GPT(model.config).cpu()  # 同结构 CPU 副本
                cpu_model.load_state_dict(model.state_dict())  # 同一份权重
                cpu_model.eval()  # 关闭 dropout
                with torch.no_grad():  # 只前向
                    _, cpu_loss = cpu_model(x.cpu(), y.cpu())  # CPU 复算
                print(f"对照：同一权重同一批数据，CPU 前向 loss = {cpu_loss.item():.4f}")  # 打印对照值
            break  # 退出循环
        optimizer.zero_grad(set_to_none=True)  # 清空梯度
        loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()  # 更新参数
        if it % 10 == 0:  # 每 10 步打印
            dt = time.time() - t0  # 最近 10 步耗时
            t0 = time.time()  # 重置计时
            print(f"iter {it:5d}/{cfg['max_iters']} | sft loss {loss.item():.4f} | lr {lr:.2e} | {dt * 100:,.0f} ms/iter")  # 打印状态
        if it % cfg["eval_interval"] == 0 or it == cfg["max_iters"]:  # 定期评估保存
            losses = estimate_loss(model, ids_dict, label_dict, cfg, device, autocast_ctx)  # 评估
            print(f"iter {it:5d} | eval train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")  # 打印评估
            if math.isfinite(losses["val"]):  # 只保存有效权重
                os.makedirs(OUT_DIR, exist_ok=True)  # 确保目录存在
                sft_path = os.path.join(OUT_DIR, "model_sft.pt")  # SFT 模型单独命名，不影响 out/model.pt
                torch.save({"model": model.state_dict(), "config": model.config.__dict__, "iter": it, "val_loss": losses["val"]}, sft_path)  # 保存
                print(f"SFT 模型已保存到 {sft_path}（原预训练模型 {args.model} 保持不变）")  # 保存提示
            else:  # 验证损失异常
                print("验证 loss 非有限值，跳过保存，保留上一个有效 checkpoint")  # 保护已有 checkpoint
    print("SFT 训练完成")  # 结束提示


if __name__ == "__main__":  # 作为脚本直接运行时才执行
    main()  # 调用主流程
