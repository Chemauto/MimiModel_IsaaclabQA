# -*- coding: utf-8 -*-
"""预训练脚本：在 TinyStories 的 token 数据上做"下一 token 预测"。

用法：
  python train.py            # 默认配置，适合 GPU 机器
  python train.py --small    # 冒烟配置，CPU 几分钟能看到 loss 下降
"""

import argparse  # 命令行参数解析库
import math  # 数学库，用于余弦学习率调度
import os  # 系统接口，用于拼接路径、创建目录
import time  # 时间库，用于打印训练速度

import numpy as np  # 读取 uint16 token 二进制文件
import torch  # PyTorch 核心库
from model import GPT, GPTConfig  # 导入模型定义

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录（mimi_gpt/）
DATA_DIR = os.path.join(BASE_DIR, "data")  # 数据目录（prepare_data.py 产出）
OUT_DIR = os.path.join(BASE_DIR, "out")  # checkpoint 输出目录（已被 .gitignore 忽略）

CONFIGS = {  # 两套预设超参数：默认给 GPU，small 给 CPU 冒烟
    "default": dict(n_layer=6, n_head=6, n_embd=384, block_size=256, batch_size=64,  # 模型与批量：约 10M 参数
                    learning_rate=6e-4, min_lr=6e-5, warmup_iters=200,  # 学习率：线性预热后余弦退火到 min_lr
                    max_iters=5000, eval_interval=200, eval_iters=100, dropout=0.1),  # 优化与评估节奏
    "small": dict(n_layer=2, n_head=2, n_embd=128, block_size=128, batch_size=16,  # 缩小的模型：约 1M 参数
                  learning_rate=1e-3, min_lr=1e-4, warmup_iters=50,  # 同样采用预热 + 退火
                  max_iters=200, eval_interval=50, eval_iters=20, dropout=0.1),  # 少量迭代快速验证流程
}


def get_lr(it, cfg):  # 第 it 步的学习率：先线性预热，再余弦退火（nanoGPT 的稳定训练配方）
    if it < cfg["warmup_iters"]:  # 预热阶段：学习率从 0 线性升到目标值，避免训练初期大步长导致发散
        return cfg["learning_rate"] * (it + 1) / cfg["warmup_iters"]  # 线性上升
    if it >= cfg["max_iters"]:  # 训练结束之后
        return cfg["min_lr"]  # 停在最低学习率
    ratio = (it - cfg["warmup_iters"]) / (cfg["max_iters"] - cfg["warmup_iters"])  # 预热完成后的训练进度 0..1
    return cfg["min_lr"] + 0.5 * (cfg["learning_rate"] - cfg["min_lr"]) * (1 + math.cos(math.pi * ratio))  # 沿余弦曲线从峰值平滑降到 min_lr


def get_batch(split_data, block_size, batch_size, device):  # 从一份 token 数据中随机取一批样本
    """返回 (x, y)：x 是连续 block_size 个 token，y 是右移一位的目标（下一 token 预测）。"""
    ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))  # 每条样本的随机起点（留出余量保证能切满 block_size+1 个 token）
    x = torch.stack([torch.from_numpy(split_data[i:i + block_size].astype(np.int64)) for i in ix])  # 输入序列 (batch, block_size)
    y = torch.stack([torch.from_numpy(split_data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])  # 目标序列：整体右移一位
    return x.to(device), y.to(device)  # 搬到计算设备（cuda/cpu）


@torch.no_grad()  # 评估阶段不需要梯度
def estimate_loss(model, data_dict, cfg, device):  # 在训练/验证集上各采样若干批，估计平均 loss
    """返回 {"train": ..., "val": ...} 的平均损失字典。"""
    model.eval()  # 切到评估模式（关闭 dropout）
    out = {}  # 结果容器
    for split, data in data_dict.items():  # 遍历训练集和验证集
        losses = torch.zeros(cfg["eval_iters"])  # 每批的 loss 存入该张量
        for k in range(cfg["eval_iters"]):  # 取多批求平均，降低随机波动
            X, Y = get_batch(data, cfg["block_size"], cfg["batch_size"], device)  # 取一批数据
            _, loss = model(X, Y)  # 前向计算损失
            losses[k] = loss.item()  # 记录该批 loss
        out[split] = losses.mean().item()  # 平均后保存
    model.train()  # 切回训练模式（恢复 dropout）
    return out  # 返回结果


def main():  # 主流程
    parser = argparse.ArgumentParser(description="预训练 mimi_gpt")  # 命令行入口
    parser.add_argument("--small", action="store_true", help="使用 CPU 冒烟配置")  # 开关：小配置
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")  # 默认自动选
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖配置中的 batch_size")  # 显存不够或排查 GPU 兼容性时用
    args = parser.parse_args()  # 解析参数
    cfg = CONFIGS["small" if args.small else "default"]  # 选定超参数组
    if args.batch_size is not None:  # 命令行显式指定了 batch_size 时覆盖默认值
        cfg = dict(cfg)  # 复制一份再改，避免污染全局配置字典
        cfg["batch_size"] = args.batch_size  # 应用覆盖值
    device = args.device  # 计算设备
    torch.manual_seed(1337)  # 固定随机种子，保证可复现

    train_data = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")  # 内存映射读取训练 token（不占内存，按需读盘）
    val_data = np.memmap(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")  # 同上，验证集
    print(f"训练集 {len(train_data):,} tokens | 验证集 {len(val_data):,} tokens | 设备 {device}")  # 打印数据规模

    model = GPT(GPTConfig(n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],  # 按配置构建模型
                          block_size=cfg["block_size"], vocab_size=8192, dropout=cfg["dropout"])).to(device)  # 搬到计算设备
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")  # 打印参数量

    decay, no_decay = [], []  # 参数分组：矩阵权重做 weight decay，LayerNorm/偏置不做
    for p in model.parameters():  # 遍历全部参数
        (decay if p.dim() >= 2 else no_decay).append(p)  # 按 ndim 分组（嵌入和线性层是 2D；LN 权重、偏置是 1D）
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],  # 分组 decay（nanoGPT 配方：对 1D 参数衰减会破坏训练稳定性）
                                  lr=cfg["learning_rate"], betas=(0.9, 0.95))  # AdamW 优化器
    data_dict = {"train": train_data, "val": val_data}  # 打包给评估函数用
    model.train()  # 训练模式
    t0 = time.time()  # 记时起点
    for it in range(1, cfg["max_iters"] + 1):  # 主训练循环
        lr = get_lr(it, cfg)  # 按预热 + 余弦退火计划计算当前学习率
        for g in optimizer.param_groups:  # 应用到所有参数组
            g["lr"] = lr  # 设置本步学习率
        x, y = get_batch(train_data, cfg["block_size"], cfg["batch_size"], device)  # 取一批训练数据
        _, loss = model(x, y)  # 前向计算 loss
        if not torch.isfinite(loss):  # loss 变成 NaN/Inf 说明训练已经发散
            print(f"iter {it}: loss = {loss.item()}，训练发散，提前停止（不保存本次权重）")  # 明确提示发散位置，便于定位原因
            break  # 立即退出，避免带着 NaN 权重继续空跑
        optimizer.zero_grad(set_to_none=True)  # 清空上一轮梯度
        loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪到 1.0，防止偶发大梯度破坏训练
        optimizer.step()  # 更新参数
        if it % 10 == 0:  # 每 10 步打印一次训练状态
            dt = time.time() - t0  # 最近 10 步耗时
            t0 = time.time()  # 重置计时
            print(f"iter {it:5d}/{cfg['max_iters']} | train loss {loss.item():.4f} | lr {lr:.2e} | {dt * 100:,.0f} ms/iter")  # 打印 loss、学习率和速度
        if it % cfg["eval_interval"] == 0 or it == cfg["max_iters"]:  # 定期评估并保存
            losses = estimate_loss(model, data_dict, cfg, device)  # 评估训练/验证 loss
            print(f"iter {it:5d} | eval train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")  # 打印评估结果
            if math.isfinite(losses["val"]):  # 只保存权重有效的 checkpoint
                os.makedirs(OUT_DIR, exist_ok=True)  # 确保输出目录存在
                torch.save({"model": model.state_dict(), "config": model.config.__dict__, "iter": it, "val_loss": losses["val"]},  # 保存权重+配置+进度
                           os.path.join(OUT_DIR, "model.pt"))  # checkpoint 路径
                print(f"模型已保存到 {os.path.join(OUT_DIR, 'model.pt')}")  # 保存提示
            else:  # 验证 loss 非有限值，说明权重已损坏
                print("验证 loss 非有限值，跳过保存，保留上一个有效 checkpoint")  # 防止用坏权重覆盖好 checkpoint
    print("训练完成")  # 结束提示


if __name__ == "__main__":  # 作为脚本直接运行时才执行
    main()  # 调用主流程
