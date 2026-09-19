# -*- coding: utf-8 -*-
"""续写脚本：加载训练好的模型，输入一个故事开头，让模型接着写下去。

用法示例：
  python sample.py --prompt "Once upon a time"
  python sample.py --prompt "One day, a little dog" --max_new_tokens 500 --temperature 0.7
"""

import argparse  # 命令行参数解析库
from pathlib import Path  # 路径操作库

import torch  # PyTorch 核心库
from tokenizers import Tokenizer  # tokenizers 库：加载训练好的 BPE 分词器
from model import GPT, GPTConfig  # 模型定义

BASE_DIR = Path(__file__).resolve().parent  # 脚本所在目录（mimi_gpt/）


def main():  # 主流程：加载模型 → 编码开头 → 生成 → 解码打印
    parser = argparse.ArgumentParser(description="用训练好的 mimi_gpt 续写故事")  # 命令行入口
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="故事开头")  # 提示文本
    parser.add_argument("--max_new_tokens", type=int, default=300, help="最多续写多少个 token")  # 生成长度上限
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度：<1 更保守，>1 更发散")  # 温度参数
    parser.add_argument("--top_k", type=int, default=50, help="只从概率最高的 k 个 token 中采样")  # top-k 截断
    parser.add_argument("--model", type=str, default=str(BASE_DIR / "out" / "model.pt"), help="checkpoint 路径")  # 模型文件
    args = parser.parse_args()  # 解析参数

    device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择计算设备
    ckpt = torch.load(args.model, map_location=device)  # 读取 checkpoint（含模型配置和权重）
    model = GPT(GPTConfig(**ckpt["config"])).to(device)  # 按保存时的配置重建同结构模型
    model.load_state_dict(ckpt["model"])  # 加载训练好的权重
    if not all(torch.isfinite(p).all() for p in model.parameters()):  # 检查权重里是否混入 NaN/Inf
        raise SystemExit("checkpoint 权重包含 NaN/Inf：模型在训练中已发散，请删除 out/model.pt 后重新训练")  # 直接给出原因和处理方式，而不是在采样时报底层错误
    model.eval()  # 切到评估模式（关闭 dropout）
    print(f"已加载 {args.model}（iter {ckpt['iter']}，val loss {ckpt['val_loss']:.4f}）")  # 打印模型信息

    tok = Tokenizer.from_file(str(BASE_DIR / "data" / "tokenizer.json"))  # 加载分词器
    eot_id = tok.token_to_id("<|endoftext|>")  # 结束符的 id，采样到它就停止生成
    start_ids = tok.encode(args.prompt).ids  # 把故事开头编码成 token id 序列
    x = torch.tensor([start_ids], dtype=torch.long, device=device)  # 增加 batch 维后转为张量 (1, T)
    y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, eos_id=eot_id)  # 自回归续写
    print("-" * 60)  # 分隔线
    text = tok.decode(y[0].tolist())  # 解码完整文本（含用户给的开头）
    print(text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore"))  # 丢弃在多字节字符中间被截断产生的无效字节，避免打印乱码
    print("-" * 60)  # 分隔线


if __name__ == "__main__":  # 作为脚本直接运行时才执行
    main()  # 调用主流程
