# -*- coding: utf-8 -*-
"""SFT 数据准备：下载 TinyStories-instruct，打包成「指令 → 故事」格式的微调样本。

每个样本的文本形态（与预训练共用同一套分词器，不新增任何特殊 token）：
  Instructions: <指令文本>\nStory: <故事><|endoftext|>

损失掩码：Instructions/Story 提示部分 label 置 -100（不计算损失），只在故事部分学习。

运行后在 data/ 目录产出（不影响 prepare_data.py 的产物）：
  sft_train_ids.bin / sft_train_labels.bin   训练集：input_ids 与 labels，各 (n, 256) 的 int16
  sft_val_ids.bin   / sft_val_labels.bin     验证集：同上
"""

import argparse  # 命令行参数解析库
from pathlib import Path  # 路径操作库

from datasets import load_dataset  # HuggingFace datasets 库，流式下载数据集
from tokenizers import Tokenizer  # 加载与预训练同一个 BPE 分词器

ROOT_DIR = Path(__file__).resolve().parent.parent  # 项目根目录 mimi_gpt/（本脚本位于 sft/ 子目录）
DATA_DIR = ROOT_DIR / "data"  # 数据目录（与预训练阶段共享，已被 .gitignore 忽略）


def build_instruction_text(item):  # 从一条数据里拼出指令文本
    """优先使用数据集自带的完整指令字段，否则把零散字段拼装成指令。"""
    for key in ("task", "instructions"):  # 若数据集直接提供完整指令文本字段，优先使用
        if item.get(key):  # 字段存在且非空
            return str(item[key]).strip()  # 返回去空白后的指令文本
    parts = []  # 否则逐个收集零散的指令字段
    if item.get("summary"):  # 情节摘要
        parts.append("Summary: " + str(item["summary"]).strip())  # 摘要作为主要指令
    if item.get("features"):  # 故事特征要求（如必须包含对话）
        parts.append("Features: " + str(item["features"]).strip())  # 特征作为附加指令
    if item.get("words"):  # 必须使用的词汇
        parts.append("Words: " + str(item["words"]).strip())  # 词汇约束作为附加指令
    return "\n".join(parts)  # 多行拼接成完整指令文本


def build_sample(tok, item, max_len):  # 把一条原始数据编码成 (ids, labels) 样本
    """返回 (input_ids, labels)；样本缺失或超出 max_len 时返回 None。"""
    story = str(item.get("story", "")).strip()  # 取出故事正文
    if not story:  # 缺故事的数据无法使用
        return None  # 跳过
    instr = build_instruction_text(item)  # 拼出指令文本
    prompt = f"Instructions: {instr}\nStory:"  # 提示模板（模型据此知道任务和背景）
    prompt_ids = tok.encode(prompt).ids  # 提示部分编码（这些位置不学习）
    resp_ids = tok.encode(" " + story + "<|endoftext|>").ids  # 回答部分编码：故事 + 结束符（这些位置要学习）
    if len(prompt_ids) + len(resp_ids) > max_len:  # 总长超出模型上下文窗口
        return None  # 跳过（保证不截断，训练样本都是完整样本）
    ids = prompt_ids + resp_ids  # 模型的完整输入序列
    labels = [-100] * len(prompt_ids) + resp_ids  # 提示位置置 -100（交叉熵默认忽略），故事位置保留真值
    return ids, labels  # 返回编码结果


def pad_and_write(samples, ids_path, labels_path, max_len):  # 补齐到统一长度并写成二进制
    """把变长样本补零到 max_len，写成 (n, max_len) 的 int16 二进制文件。"""
    ids_rows, label_rows = [], []  # 收集所有行
    for ids, labels in samples:  # 遍历每条样本
        pad = max_len - len(ids)  # 需要补齐的长度
        ids_rows.append(ids + [0] * pad)  # 输入用 <|endoftext|>(id=0) 补齐
        label_rows.append(labels + [-100] * pad)  # 补齐位置不计算损失
    import numpy as np  # 延迟导入：写文件时才需要
    np.array(ids_rows, dtype=np.int16).tofile(ids_path)  # 词表 8k、-100 都在 int16 范围内
    np.array(label_rows, dtype=np.int16).tofile(labels_path)  # labels 同样写成 int16


def main():  # 主流程：下载 → 编码打包 → 落盘
    parser = argparse.ArgumentParser(description="准备 TinyStories-instruct SFT 数据")  # 命令行入口
    parser.add_argument("--max_train_samples", type=int, default=50_000, help="训练样本条数上限")  # 训练规模
    parser.add_argument("--max_val_samples", type=int, default=500, help="验证样本条数上限")  # 验证规模
    parser.add_argument("--max_len", type=int, default=256, help="单条样本最大 token 数（须不超过模型 block_size）")  # 序列长度上限
    args = parser.parse_args()  # 解析参数

    tok = Tokenizer.from_file(str(DATA_DIR / "tokenizer.json"))  # 加载预训练时训好的分词器（必须与预训练一致）
    ds = load_dataset("roneneldan/TinyStories-instruct", split="train", streaming=True)  # 流式读取，只取需要的条数

    DATA_DIR.mkdir(exist_ok=True)  # 确保数据目录存在
    train_samples, val_samples = [], []  # 分别收集训练/验证样本
    skipped = 0  # 统计被跳过的条数
    for i, item in enumerate(ds.take(args.max_train_samples + args.max_val_samples)):  # 顺序取出指定条数
        if i < args.max_val_samples:  # 最前面的一部分划给验证集
            split_samples = val_samples  # 指向验证列表
        else:  # 其余划给训练集
            split_samples = train_samples  # 指向训练列表
        result = build_sample(tok, item, args.max_len)  # 编码一条样本
        if result is None:  # 缺故事或超长
            skipped += 1  # 计入跳过数
            continue  # 处理下一条
        split_samples.append(result)  # 加入对应列表
        if (i + 1) % 5000 == 0:  # 每处理 5000 条打印进度
            print(f"已处理 {i + 1} 条（训练 {len(train_samples)} / 验证 {len(val_samples)} / 跳过 {skipped}）")

    pad_and_write(train_samples, DATA_DIR / "sft_train_ids.bin", DATA_DIR / "sft_train_labels.bin", args.max_len)  # 训练集落盘
    pad_and_write(val_samples, DATA_DIR / "sft_val_ids.bin", DATA_DIR / "sft_val_labels.bin", args.max_len)  # 验证集落盘
    print(f"SFT 数据准备完成：训练 {len(train_samples)} 条，验证 {len(val_samples)} 条，跳过 {skipped} 条")  # 汇总
    if train_samples:  # 有样本时打印一条示例，方便人工检查模板和掩码
        ids, labels = train_samples[0]  # 取第一条样本
        print("---- 示例（第一条样本解码 + 掩码统计）----")  # 说明打印含义
        print(tok.decode(ids)[:200])  # 打印解码后的文本前 200 字符
        learned = sum(1 for l in labels if l != -100)  # 统计参与损失计算的位置数
        print(f"总长 {len(ids)}，其中参与损失计算的位置 {learned} 个（即故事部分）")  # 打印掩码统计


if __name__ == "__main__":  # 作为脚本直接运行时才执行
    main()  # 调用主流程
