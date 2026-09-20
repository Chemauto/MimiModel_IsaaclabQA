# -*- coding: utf-8 -*-
"""SFT 数据准备：把指令-回答语料打包成微调样本，支持 --dataset 切换两种链路。

【tinystories-instruct（默认，英文）】原始数据是按行存储的 txt，块结构为
  指令行（Features / Words / Summary）→ "Story:" → 故事正文 → <|endoftext|>
重组后编码成模板（不新增任何特殊 token）：
  Instructions: <指令文本>\nStory: <故事><|endoftext|>

【minimind（中文）】sft_t2t_mini.jsonl 的 conversations 多轮对话，拆成单轮 (问, 答)：
  问：<用户提问>\n答：<助手回答><|endoftext|>

损失掩码：提示部分 label 置 -100（不计算损失），标签相对输入右移一位（预测下一个 token）。

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


def iter_examples(ds):  # 英文链路：把按行流式读取的数据重组为 (指令文本, 故事文本)
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


def iter_chat_pairs(ds):  # 中文链路：把多轮对话拆成单轮 (问, 答) 样本
    """每条会话按轮次顺序配对：暂存提问，遇到回答即产出 (问, 答)。"""
    for row in ds:  # 逐条会话
        conv = row.get("conversations") or []  # 会话轮次列表
        pending_user = None  # 待配对的提问
        for turn in conv:  # 遍历轮次
            role = turn.get("role")  # 角色（user/assistant）
            content = (turn.get("content") or "").strip()  # 轮次内容
            if not content:  # 空内容跳过
                continue  # 下一轮
            if role == "user":  # 提问轮
                pending_user = content  # 暂存，等待配对回答
            elif role == "assistant" and pending_user:  # 回答轮且有配对提问
                yield pending_user, content  # 产出 (问, 答)
                pending_user = None  # 清空，后续轮重新配对


def build_sample(tok, prompt, response, max_len):  # 把一条 (提示, 回答) 编码成训练样本
    """返回 (input_ids, labels, truncated)；提示过长返回 None，回答过长截断兜底。"""
    prompt_ids = tok.encode(prompt).ids  # 提示部分编码（这些位置不学习）
    budget = max_len - len(prompt_ids) - 1  # 回答可用 token 数（预留 1 位给结束符）
    if budget < 16:  # 提示就把窗口占满了，剩余空间写不出有意义的回答
        return None  # 跳过
    resp_ids = tok.encode(response).ids  # 回答部分编码
    truncated = len(resp_ids) > budget  # 是否需要截断
    if truncated:  # 截断兜底（保留回答开头，丢弃尾部），避免样本被整条丢弃
        resp_ids = resp_ids[:budget]  # 截断到可用空间
    resp_ids = resp_ids + [tok.token_to_id("<|endoftext|>")]  # 末尾补结束符，模型学会"答到这里为止"
    ids = prompt_ids + resp_ids  # 模型的完整输入序列
    # 标签必须右移一位（语言模型的本职是预测"下一个"token）：
    #   位置 t 的标签是 ids[t+1]；提示内部位置不学习（-100）；
    #   提示的最后一个位置负责产出回答的第一个 token，回答的最后一个位置负责产出结束符。
    labels = [-100] * len(ids)  # 先全部置为忽略
    for t in range(len(prompt_ids) - 1, len(ids) - 1):  # 从提示末位遍历到倒数第二位
        labels[t] = ids[t + 1]  # 标签指向下一个 token
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


def main():  # 主流程：下载数据 → 按模板编码打包 → 落盘
    parser = argparse.ArgumentParser(description="准备 SFT 微调数据")  # 命令行入口
    parser.add_argument("--dataset", type=str, default="tinystories-instruct",  # 数据集选择
                        choices=["tinystories-instruct", "minimind"],  # 英文故事 / 中文对话两条链路
                        help="tinystories-instruct=英文按摘要写故事（默认），minimind=中文问答回话（需预训练也用 minimind 语料）")  # 说明
    parser.add_argument("--max_train_samples", type=int, default=50_000, help="训练样本条数上限")  # 训练规模
    parser.add_argument("--max_val_samples", type=int, default=500, help="验证样本条数上限")  # 验证规模
    parser.add_argument("--max_len", type=int, default=256, help="单条样本最大 token 数（须不超过模型 block_size）")  # 序列长度上限
    args = parser.parse_args()  # 解析参数

    tok = Tokenizer.from_file(str(DATA_DIR / "tokenizer.json"))  # 加载预训练时训好的分词器（必须与预训练一致）
    if args.dataset == "minimind":  # 中文问答回话链路
        ds = load_dataset("jingyaogong/minimind_dataset", data_files="sft_t2t_mini.jsonl",  # minimind SFT 数据
                          split="train", streaming=True)  # 流式读取
        example_iter = iter_chat_pairs(ds)  # 拆成 (问, 答) 迭代器
    else:  # 英文按摘要写故事链路
        ds = load_dataset("roneneldan/TinyStories-instruct", split="train", streaming=True)  # 流式读取原始 txt
        example_iter = ((pick_instruction(instr), story) for instr, story in iter_examples(ds))  # (摘要, 故事) 迭代器

    train_samples, val_samples = [], []  # 分别收集训练/验证样本
    skipped, truncated = 0, 0  # 统计：跳过条数、回答被截断条数
    processed = 0  # 已处理的完整样本数
    for source_text, response_text in example_iter:  # 逐条产出 (提示素材, 回答正文)
        if args.dataset == "minimind":  # 中文模板：问/答
            prompt, response = f"问：{source_text}\n答：", response_text  # 拼提示与回答
        else:  # 英文模板：Instructions/Story
            prompt, response = f"Instructions: {source_text}\nStory:", " " + response_text  # 拼提示与回答（回答带前导空格）
        if processed == 0:  # 打印第一条的提示与回答开头，便于人工核对模板
            print("首条提示:", prompt[:150])  # 提示拼接结果
            print("首条回答开头:", response[:100])  # 回答开头
        split_samples = val_samples if len(val_samples) < args.max_val_samples else train_samples  # 前若干条划给验证集，其余给训练集
        result = build_sample(tok, prompt, response, args.max_len)  # 编码样本
        if result is None:  # 缺回答或提示过长
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
    print(f"SFT 数据准备完成：训练 {len(train_samples)} 条，验证 {len(val_samples)} 条，跳过 {skipped} 条，回答截断 {truncated} 条")  # 汇总
    if train_samples:  # 有样本时打印一条示例，方便人工检查模板和掩码
        ids, labels = train_samples[0]  # 取第一条样本
        print("---- 示例（第一条样本解码 + 掩码统计）----")  # 说明打印含义
        print(tok.decode(ids)[:200])  # 打印解码后的文本前 200 字符
        learned = sum(1 for l in labels if l != -100)  # 统计参与损失计算的位置数
        print(f"总长 {len(ids)}，其中参与损失计算的位置 {learned} 个（即回答部分）")  # 打印掩码统计


if __name__ == "__main__":  # 作为脚本直接运行时才执行
    main()  # 调用主流程
