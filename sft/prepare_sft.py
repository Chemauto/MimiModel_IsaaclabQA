# -*- coding: utf-8 -*-
"""SFT 数据准备：下载 TinyStories-instruct，打包成「指令 → 故事」格式的微调样本。

原始数据是按行存储的 txt（流式读取后每行一个 'text' 字段），每条样本的块结构为：
  指令行（Features / Words / Summary，顺序不定，可有可无）
  Story:
  空行 + 故事正文若干行
  <|endoftext|>
本脚本把行重组为 (指令, 故事) 对，再编码成统一模板（不新增任何特殊 token）：
  Instructions: <指令文本>\nStory: <故事><|endoftext|>
损失掩码：提示部分 label 置 -100（不计算损失），只在故事部分学习。

运行后在 data/ 目录产出（不影响 prepare_data.py 的产物）：
  sft_train_ids.bin / sft_train_labels.bin   训练集：input_ids 与 labels，各 (n, max_len) 的 int16
  sft_val_ids.bin   / sft_val_labels.bin     验证集：同上
"""

import argparse  # 命令行参数解析库
from pathlib import Path  # 路径操作库

import numpy as np  # 把 token id 写成二进制文件
from datasets import load_dataset  # HuggingFace datasets 库，流式下载数据集
from tokenizers import Tokenizer  # 加载与预训练同一个 BPE 分词器

ROOT_DIR = Path(__file__).resolve().parent.parent  # 项目根目录 mimi_gpt/（本脚本位于 sft/ 子目录）
DATA_DIR = ROOT_DIR / "data"  # 数据目录（与预训练阶段共享，已被 .gitignore 忽略）


def iter_examples(ds):  # 把按行流式读取的数据重组为 (指令文本, 故事文本)
    """逐行遍历数据流，按块结构切分并产出完整的 (指令, 故事) 样本。"""
    instr_lines, story_lines, in_story = [], [], False  # 解析状态：指令行缓存、故事行缓存、是否已进入故事区
    for row in ds:  # 逐行遍历流式数据
        line = row["text"].strip()  # 取出该行文本并去首尾空白
        if line == "<|endoftext|>":  # 样本结束标记
            if story_lines:  # 收集到了故事正文才产出（跳过脏块）
                yield "\n".join(instr_lines), " ".join(story_lines)  # 产出 (指令块, 故事正文)
            instr_lines, story_lines, in_story = [], [], False  # 重置状态，开始下一条
        elif in_story:  # 已进入故事区
            if line:  # 收集故事正文（跳过空行）
                story_lines.append(line)  # 追加到故事缓存
        elif line.startswith("Story:"):  # "Story:" 行是故事区的开始标记
            in_story = True  # 切换到故事收集状态
        elif line:  # 还在指令区，收集非空指令行
            instr_lines.append(line)  # 追加到指令缓存


def pick_instruction(instr_text):  # 从指令块里挑出最紧凑的指令文本
    """摘要行最能概括任务且最短，能给故事留出最多的生成空间。"""
    for line in instr_text.splitlines():  # 逐行查找
        if line.startswith("Summary:"):  # 找到摘要行
            return line.strip()  # 直接采用摘要作为指令
    return instr_text  # 没有摘要时用全部指令行


def build_sample(tok, instr_text, story, max_len):  # 把一条 (指令, 故事) 编码成训练样本
    """返回 (input_ids, labels, truncated)；提示过长返回 None，故事过长截断兜底。"""
    instr = pick_instruction(instr_text)  # 选出紧凑指令文本
    prompt = f"Instructions: {instr}\nStory:"  # 提示模板（模型据此知道任务和背景）
    prompt_ids = tok.encode(prompt).ids  # 提示部分编码（这些位置不学习）
    budget = max_len - len(prompt_ids) - 1  # 故事可用 token 数（预留 1 位给结束符）
    if budget < 16:  # 提示就把窗口占满了，剩余空间写不出有意义的故事
        return None  # 跳过
    resp_ids = tok.encode(" " + story).ids  # 回答部分编码：故事正文
    truncated = len(resp_ids) > budget  # 是否需要截断
    if truncated:  # 截断兜底（保留故事开头，丢弃尾部），避免样本被整条丢弃
        resp_ids = resp_ids[:budget]  # 截断到可用空间
    resp_ids = resp_ids + [tok.token_to_id("<|endoftext|>")]  # 末尾补结束符，模型学会"写到这里为止"
    ids = prompt_ids + resp_ids  # 模型的完整输入序列
    labels = [-100] * len(prompt_ids) + resp_ids  # 提示位置置 -100（交叉熵默认忽略），故事位置保留真值
    return ids, labels, truncated  # 返回编码结果与截断标记


def pad_and_write(samples, ids_path, labels_path, max_len):  # 补齐到统一长度并写成二进制
    """把变长样本补零到 max_len，写成 (n, max_len) 的 int16 二进制文件。"""
    ids_rows, label_rows = [], []  # 收集所有行
    for ids, labels in samples:  # 遍历每条样本
        pad = max_len - len(ids)  # 需要补齐的长度
        ids_rows.append(ids + [0] * pad)  # 输入用 <|endoftext|>(id=0) 补齐
        label_rows.append(labels + [-100] * pad)  # 补齐位置不计算损失
    np.array(ids_rows, dtype=np.int16).tofile(ids_path)  # 词表 8k、-100 都在 int16 范围内
    np.array(label_rows, dtype=np.int16).tofile(labels_path)  # labels 同样写成 int16


def main():  # 主流程：下载 → 重组 → 编码打包 → 落盘
    parser = argparse.ArgumentParser(description="准备 TinyStories-instruct SFT 数据")  # 命令行入口
    parser.add_argument("--max_train_samples", type=int, default=50_000, help="训练样本条数上限")  # 训练规模
    parser.add_argument("--max_val_samples", type=int, default=500, help="验证样本条数上限")  # 验证规模
    parser.add_argument("--max_len", type=int, default=256, help="单条样本最大 token 数（须不超过模型 block_size）")  # 序列长度上限
    args = parser.parse_args()  # 解析参数

    tok = Tokenizer.from_file(str(DATA_DIR / "tokenizer.json"))  # 加载预训练时训好的分词器（必须与预训练一致）
    ds = load_dataset("roneneldan/TinyStories-instruct", split="train", streaming=True)  # 流式读取原始 txt

    DATA_DIR.mkdir(exist_ok=True)  # 确保数据目录存在
    train_samples, val_samples = [], []  # 分别收集训练/验证样本
    skipped, truncated = 0, 0  # 统计：跳过条数（缺故事/提示过长）、故事被截断条数
    processed = 0  # 已处理的完整样本数
    for instr_text, story in iter_examples(ds):  # 逐条产出重组后的样本
        if processed == 0:  # 打印第一条的指令与故事开头，便于人工核对解析结果
            print("首条指令:", pick_instruction(instr_text)[:150])  # 指令拼接结果
            print("首条故事开头:", story[:100])  # 故事开头
        split_samples = val_samples if len(val_samples) < args.max_val_samples else train_samples  # 前若干条划给验证集，其余给训练集
        result = build_sample(tok, instr_text, story, args.max_len)  # 编码样本
        if result is None:  # 缺故事或提示过长
            skipped += 1  # 计入跳过数
        else:  # 编码成功
            ids, labels, was_truncated = result  # 解包编码结果
            truncated += int(was_truncated)  # 累计截断数
            split_samples.append((ids, labels))  # 加入对应列表
        processed += 1  # 完整样本计数
        if processed % 5000 == 0:  # 每处理 5000 条打印进度
            print(f"已处理 {processed} 条（训练 {len(train_samples)} / 验证 {len(val_samples)} / 跳过 {skipped} / 截断 {truncated}）")
        if len(train_samples) >= args.max_train_samples:  # 训练集收满即停（流式数据无需读完）
            break  # 退出循环

    pad_and_write(train_samples, DATA_DIR / "sft_train_ids.bin", DATA_DIR / "sft_train_labels.bin", args.max_len)  # 训练集落盘
    pad_and_write(val_samples, DATA_DIR / "sft_val_ids.bin", DATA_DIR / "sft_val_labels.bin", args.max_len)  # 验证集落盘
    print(f"SFT 数据准备完成：训练 {len(train_samples)} 条，验证 {len(val_samples)} 条，跳过 {skipped} 条，故事截断 {truncated} 条")  # 汇总
    if train_samples:  # 有样本时打印一条示例，方便人工检查模板和掩码
        ids, labels = train_samples[0]  # 取第一条样本
        print("---- 示例（第一条样本解码 + 掩码统计）----")  # 说明打印含义
        print(tok.decode(ids)[:200])  # 打印解码后的文本前 200 字符
        learned = sum(1 for l in labels if l != -100)  # 统计参与损失计算的位置数
        print(f"总长 {len(ids)}，其中参与损失计算的位置 {learned} 个（即故事部分）")  # 打印掩码统计


if __name__ == "__main__":  # 作为脚本直接运行时才执行
    main()  # 调用主流程
