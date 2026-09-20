# -*- coding: utf-8 -*-
"""DPO 训练脚本：从 SFT 模型出发，用偏好对做直接偏好优化。

DPO 的思想：不再单独训练奖励模型，而是把「偏好」写成策略与参考模型的对数似然差：
  loss = -logsigmoid( beta * [ (logπ(y+w|x) - logπref(y+w|x)) - (logπ(y-l|x) - logπref(y-l|x)) ] )
其中 y+w 是 chosen（好回答）、y-l 是 rejected（差回答）。训练推动策略相对参考模型
更偏向 chosen、远离 rejected。不修改任何既有产物：从 out/model_sft.pt 读取，保存到 out/model_dpo.pt。

用法：
  python dpo/train_dpo.py                        # 默认配置（GPU 300 iter，几分钟）
  python dpo/train_dpo.py --beta 0.1 --learning_rate 1e-5
"""

import argparse  # 命令行参数解析库
import math  # 数学库（余弦调度）
import os  # 路径与目录
import sys  # 系统路径操作
import time  # 计时

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # 必须在 import torch 之前设置（与 train.py 相同的稳定化措施）

import numpy as np  # 读取 int16 样本
import torch  # PyTorch
import torch.nn.functional as F  # logsigmoid 等函数

torch.use_deterministic_algorithms(True, warn_only=True)  # 确定性内核（与 train.py 相同）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录 mimi_gpt/
sys.path.insert(0, ROOT_DIR)  # 导入根目录的 model.py
from model import GPT, GPTConfig  # 模型定义

DATA_DIR = os.path.join(ROOT_DIR, "data")  # 数据目录
OUT_DIR = os.path.join(ROOT_DIR, "out")  # checkpoint 输出目录

DPO_CONFIG = dict(batch_size=16, learning_rate=1e-5, min_lr=1e-6,  # DPO 学习率要比 SFT 更小（1e-5 级），否则容易训崩
                  beta=0.1, warmup_iters=20, max_iters=300, eval_interval=100, eval_iters=30)  # beta 越大偏好越强、越保守


def get_lr(it, cfg):  # 线性预热 + 余弦退火（与其他阶段同一套调度）
    if it < cfg["warmup_iters"]:  # 预热阶段
        return cfg["learning_rate"] * (it + 1) / cfg["warmup_iters"]  # 线性上升
    if it >= cfg["max_iters"]:  # 结束后
        return cfg["min_lr"]  # 停在最低值
    ratio = (it - cfg["warmup_iters"]) / (cfg["max_iters"] - cfg["warmup_iters"])  # 进度 0..1
    return cfg["min_lr"] + 0.5 * (cfg["learning_rate"] - cfg["min_lr"]) * (1 + math.cos(math.pi * ratio))  # 余弦下降


def get_batch(ids_data, mask_data, batch_size, device):  # 随机取一批序列与回答掩码
    n = ids_data.shape[0]  # 样本总数
    ix = torch.randint(n, (batch_size,))  # 随机行号
    x = torch.stack([torch.from_numpy(ids_data[i].astype(np.int64)) for i in ix])  # 序列
    m = torch.stack([torch.from_numpy(mask_data[i].astype(np.float32)) for i in ix])  # 回答区掩码
    return x.to(device), m.to(device)  # 搬到设备


def seq_logp(model, x, mask, autocast_ctx):  # 计算模型对回答区的对数似然 Σ logπ(token)
    """返回 (B,) 每条序列在回答区（mask=1 的目标位置）的对数似然和。"""
    with autocast_ctx:  # 与整体精度策略一致
        logits, _ = model(x)  # (B, T, V)
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)  # 位置 t 的分布预测 x[t+1]，转 fp32 求对数更稳定
    tgt = x[:, 1:].unsqueeze(-1)  # 目标：右移一位的真实 token
    tok_logp = torch.gather(logp, -1, tgt).squeeze(-1)  # 取出真实 token 的对数概率 (B, T-1)
    resp = mask[:, 1:]  # 目标位置属于回答区（含结束符）才计入
    return (tok_logp * resp).sum(-1)  # 序列级对数似然


@torch.no_grad()  # 参考模型与评估都不需要梯度
def eval_metrics(policy, ref, ids_dict, mask_dict, cfg, device, autocast_ctx):  # 在验证集上估计 DPO 指标
    """返回验证集上的平均 reward accuracy 与 chosen/rejected 的对数似然差均值。"""
    policy.eval()  # 评估模式
    accs, margins = [], []  # 指标收集
    for _ in range(cfg["eval_iters"]):  # 多批平均
        xc, mc = get_batch(ids_dict["val"], mask_dict["val"], cfg["batch_size"], device)  # chosen 批
        xr, mr = get_batch(ids_dict["val"], mask_dict["val"], cfg["batch_size"], device)  # rejected 批
        pol_c = seq_logp(policy, xc, mc, autocast_ctx)  # 策略对 chosen 的对数似然
        pol_r = seq_logp(policy, xr, mr, autocast_ctx)  # 策略对 rejected 的对数似然
        ref_c = seq_logp(ref, xc, mc, autocast_ctx)  # 参考对 chosen 的对数似然
        ref_r = seq_logp(ref, xr, mr, autocast_ctx)  # 参考对 rejected 的对数似然
        logits = cfg["beta"] * ((pol_c - ref_c) - (pol_r - ref_r))  # DPO 的隐式奖励差
        accs.append((logits > 0).float().mean().item())  # 隐式奖励排序正确率
        margins.append(((pol_c - pol_r).mean() - (ref_c - ref_r).mean()).item())  # 边际变化
    policy.train()  # 切回训练模式
    return sum(accs) / len(accs), sum(margins) / len(margins)  # 平均值


def main():  # 主流程
    parser = argparse.ArgumentParser(description="DPO 后训练 mimi_gpt")  # 命令行入口
    parser.add_argument("--model", type=str, default=os.path.join(OUT_DIR, "model_sft.pt"), help="SFT checkpoint（只读，作为策略与参考的共同起点）")  # 初始权重
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")  # 设备
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖偏好对批大小")  # 覆盖
    parser.add_argument("--max_iters", type=int, default=None, help="覆盖训练步数")  # 覆盖
    parser.add_argument("--beta", type=float, default=None, help="覆盖 DPO 的 beta")  # 覆盖
    parser.add_argument("--learning_rate", type=float, default=None, help="覆盖学习率")  # 覆盖
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp32"], help="训练精度：auto 在 GPU 上用 bf16 混合")  # 精度
    args = parser.parse_args()  # 解析

    cfg = dict(DPO_CONFIG)  # 复制配置
    if args.batch_size is not None:  # 命令行覆盖
        cfg["batch_size"] = args.batch_size  # 应用
    if args.max_iters is not None:  # 覆盖步数
        cfg["max_iters"] = args.max_iters  # 应用
    if args.beta is not None:  # 覆盖 beta
        cfg["beta"] = args.beta  # 应用
    if args.learning_rate is not None:  # 覆盖学习率
        cfg["learning_rate"] = args.learning_rate  # 应用

    device = args.device  # 设备
    use_bf16 = args.dtype == "bf16" or (args.dtype == "auto" and device.startswith("cuda"))  # GPU 默认 bf16
    autocast_ctx = torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=use_bf16)  # 混合精度上下文
    torch.manual_seed(1337)  # 固定种子

    ckpt = torch.load(args.model, map_location="cpu")  # 读取 SFT checkpoint
    policy = GPT(GPTConfig(**ckpt["config"])).to(device)  # 策略模型（要训练）
    policy.load_state_dict(ckpt["model"])  # 载入 SFT 权重
    ref = GPT(GPTConfig(**ckpt["config"])).to(device)  # 参考模型（冻结）
    ref.load_state_dict(ckpt["model"])  # 同一份起点权重
    ref.eval()  # 参考模型永远评估模式
    for p in ref.parameters():  # 遍历参考模型参数
        p.requires_grad_(False)  # 全部冻结
    print(f"已加载 SFT 模型 {args.model}（val loss {ckpt['val_loss']:.4f}）作为策略与参考 | 精度 {'bf16 混合' if use_bf16 else 'fp32'}")  # 打印

    block = ckpt["config"]["block_size"]  # 序列长度
    ids_c = np.memmap(os.path.join(DATA_DIR, "dpo_chosen_ids.bin"), dtype=np.int16, mode="r").reshape(-1, block)  # chosen 序列
    mask_c = np.memmap(os.path.join(DATA_DIR, "dpo_chosen_mask.bin"), dtype=np.int16, mode="r").reshape(-1, block)  # chosen 掩码
    ids_r = np.memmap(os.path.join(DATA_DIR, "dpo_rejected_ids.bin"), dtype=np.int16, mode="r").reshape(-1, block)  # rejected 序列
    mask_r = np.memmap(os.path.join(DATA_DIR, "dpo_rejected_mask.bin"), dtype=np.int16, mode="r").reshape(-1, block)  # rejected 掩码
    assert len(ids_c) == len(ids_r), "chosen 与 rejected 的偏好对数量不一致，请重新运行 prepare_dpo.py"  # 一致性检查
    print(f"DPO 偏好对 {len(ids_c):,} 条")  # 打印规模

    decay, no_decay = [], []  # 只对策略模型建优化器（参考模型冻结）
    for p in policy.parameters():  # 遍历参数
        (decay if p.dim() >= 2 else no_decay).append(p)  # 按 ndim 分组
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],  # 分组 decay
                                  lr=cfg["learning_rate"], betas=(0.9, 0.95))  # AdamW

    ids_dict = {"val": ids_r, "train": ids_c}  # 占位命名（评估里成对取 rejected/chosen，见 eval_metrics）
    mask_dict = {"val": mask_r, "train": mask_c}  # 同上
    policy.train()  # 训练模式
    t0 = time.time()  # 计时
    for it in range(1, cfg["max_iters"] + 1):  # 主循环
        lr = get_lr(it, cfg)  # 学习率
        for g in optimizer.param_groups:  # 应用学习率
            g["lr"] = lr  # 设置
        xc, mc = get_batch(ids_c, mask_c, cfg["batch_size"], device)  # chosen 批
        xr, mr = get_batch(ids_r, mask_r, cfg["batch_size"], device)  # rejected 批
        pol_c = seq_logp(policy, xc, mc, autocast_ctx)  # 策略 chosen 对数似然
        pol_r = seq_logp(policy, xr, mr, autocast_ctx)  # 策略 rejected 对数似然
        with torch.no_grad():  # 参考模型不回传梯度
            ref_c = seq_logp(ref, xc, mc, autocast_ctx)  # 参考 chosen
            ref_r = seq_logp(ref, xr, mr, autocast_ctx)  # 参考 rejected
        logits = cfg["beta"] * ((pol_c - ref_c) - (pol_r - ref_r))  # 隐式奖励差 (B,)
        loss = -F.logsigmoid(logits.float()).mean()  # DPO 损失（fp32 求损失更稳）
        if not torch.isfinite(loss):  # 数值异常保护
            print(f"iter {it}: loss = {loss.item()}，异常停止（不保存）")  # 提示
            break  # 退出
        optimizer.zero_grad(set_to_none=True)  # 清梯度
        loss.backward()  # 反向
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)  # 裁剪
        optimizer.step()  # 更新策略
        if it % 10 == 0:  # 定期打印
            dt = time.time() - t0  # 耗时
            t0 = time.time()  # 重置
            acc = (logits > 0).float().mean().item()  # 本批隐式奖励正确率
            print(f"iter {it:5d}/{cfg['max_iters']} | dpo loss {loss.item():.4f} | reward acc {acc:.2f} | lr {lr:.2e} | {dt * 100:,.0f} ms/iter")  # 状态
        if it % cfg["eval_interval"] == 0 or it == cfg["max_iters"]:  # 定期评估保存
            acc, margin = eval_metrics(policy, ref, ids_dict, mask_dict, cfg, device, autocast_ctx)  # 验证集指标
            print(f"iter {it:5d} | val reward acc {acc:.3f} | val margin {margin:+.4f}")  # 打印指标
            if math.isfinite(acc):  # 有效性检查
                os.makedirs(OUT_DIR, exist_ok=True)  # 目录
                dpo_path = os.path.join(OUT_DIR, "model_dpo.pt")  # DPO 模型独立命名
                torch.save({"model": policy.state_dict(), "config": policy.config.__dict__, "iter": it, "val_loss": acc}, dpo_path)  # 保存（val_loss 字段这里存 reward acc）
                print(f"DPO 模型已保存到 {dpo_path}（上游 SFT/预训练模型保持不变）")  # 提示
            else:  # 异常
                print("指标非有限值，跳过保存")  # 保护
    print("DPO 训练完成")  # 结束


if __name__ == "__main__":  # 作为脚本直接运行时才执行
    main()  # 调用主流程
