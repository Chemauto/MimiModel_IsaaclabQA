# -*- coding: utf-8 -*-
"""SFT 模型采样脚本：给一句指令，让微调后的模型按指令写故事。

用法示例（在项目根目录 mimi_gpt/ 下执行）：
  python sft/sample_sft.py --instructions "Write a short story about a little kitten who makes a new friend."
  python sft/sample_sft.py --instructions "Summary: A boy loses his kite in a tree, and a bird helps him get it back."
"""

import argparse  # 命令行参数解析库
from pathlib import Path  # 路径操作库
import sys  # 系统路径操作，用于把项目根加入模块搜索路径

import torch  # PyTorch 核心库
from tokenizers import Tokenizer  # 加载分词器

ROOT_DIR = Path(__file__).resolve().parent.parent  # 项目根目录 mimi_gpt/（本脚本位于 sft/ 子目录）
sys.path.insert(0, str(ROOT_DIR))  # 把项目根加入模块搜索路径，才能导入根目录的 model.py
from model import GPT, GPTConfig  # 模型定义


def main():  # 主流程：加载 SFT 模型 → 拼模板 → 生成故事部分
    parser = argparse.ArgumentParser(description="用 SFT 模型按指令写故事")  # 命令行入口
    parser.add_argument("--instructions", type=str, default="Write a short story about a little kitten who makes a new friend.", help="给模型的指令（情节摘要/特征/要求等）")  # 指令文本
    parser.add_argument("--max_new_tokens", type=int, default=215, help="最多生成多少 token（提示已占约 40，上限为 block_size）")  # 生成长度上限
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度：<1 更保守，>1 更发散")  # 温度
    parser.add_argument("--top_k", type=int, default=50, help="只从概率最高的 k 个 token 中采样")  # top-k
    parser.add_argument("--model", type=str, default=str(ROOT_DIR / "out" / "model_sft.pt"), help="SFT checkpoint 路径")  # 模型文件
    args = parser.parse_args()  # 解析参数

    device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选设备
    ckpt = torch.load(args.model, map_location=device)  # 读取 SFT checkpoint
    model = GPT(GPTConfig(**ckpt["config"])).to(device)  # 重建模型
    model.load_state_dict(ckpt["model"])  # 加载权重
    if not all(torch.isfinite(p).all() for p in model.parameters()):  # 权重完整性检查
        raise SystemExit("checkpoint 权重包含 NaN/Inf：请重新运行 train_sft.py")  # 明确报错
    model.eval()  # 评估模式
    print(f"已加载 {args.model}（iter {ckpt['iter']}，val loss {ckpt['val_loss']:.4f}）")  # 打印模型信息

    tok = Tokenizer.from_file(str(ROOT_DIR / "data" / "tokenizer.json"))  # 分词器与预训练/SFT 完全一致
    eot_id = tok.token_to_id("<|endoftext|>")  # 结束符：故事写完时模型会输出它
    prompt = f"Instructions: {args.instructions}\nStory:"  # 与 SFT 训练时完全相同的模板
    start_ids = tok.encode(prompt).ids  # 编码提示
    x = torch.tensor([start_ids], dtype=torch.long, device=device)  # (1, T) 输入张量
    budget = ckpt["config"]["block_size"] - len(start_ids)  # 上下文窗口内还能生成的 token 数
    y = model.generate(x, min(args.max_new_tokens, budget), temperature=args.temperature, top_k=args.top_k, eos_id=eot_id)  # 自回归生成
    story_ids = y[0][len(start_ids):].tolist()  # 只取新生成的部分（不含提示）
    text = tok.decode(story_ids)  # 解码成文本
    if eot_id in story_ids:  # 若生成了结束符
        text = text.split("<|endoftext|>")[0]  # 截掉结束符及其后内容
    print("-" * 60)  # 分隔线
    print("Instructions:", args.instructions)  # 回显指令
    print("-" * 60)  # 分隔线
    print(text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore"))  # 打印故事（丢弃截断产生的无效字节）
    print("-" * 60)  # 分隔线


if __name__ == "__main__":  # 作为脚本直接运行时才执行
    main()  # 调用主流程
