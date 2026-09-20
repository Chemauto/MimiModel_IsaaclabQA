# -*- coding: utf-8 -*-
"""数据准备脚本：下载 TinyStories 数据集 → 训练 BPE 分词器 → 生成训练用二进制 token 文件。

运行后在 data/ 目录产出：
  corpus_train.txt / corpus_val.txt  纯文本语料（故事之间用 <|endoftext|> 分隔）
  tokenizer.json                     训练好的 ByteLevel BPE 分词器
  train.bin / val.bin                token id 序列（uint16，训练脚本随机切片读取）
"""

import argparse  # 命令行参数解析库
from pathlib import Path  # 面向对象的路径操作库

import numpy as np  # 数值计算库，用于把 token id 写成二进制文件
from datasets import load_dataset  # HuggingFace datasets 库，用于流式下载数据集
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders  # tokenizers 库：BPE 分词器及训练组件

BASE_DIR = Path(__file__).resolve().parent  # 脚本所在目录（mimi_gpt/）
DATA_DIR = BASE_DIR / "data"  # 数据目录：mimi_gpt/data/（已被 .gitignore 忽略，不会进 git）

DATASETS = {  # 可选预训练语料注册表：--dataset 切换
    "tinystories": dict(  # 英文童话故事（原默认，教学最稳）
        hf_id="roneneldan/TinyStories",  # HuggingFace 数据集 id
        default_train=100_000, default_val=2_000),  # 推荐条数（条 = 一个故事）
    "minimind": dict(  # minimind 项目中文混合语料（中英、对话式短文本）
        hf_id="jingyaogong/minimind_dataset",  # HuggingFace 数据集 id
        data_file="pretrain_t2t_mini.jsonl",  # 精简版预训练文件（几百 MB）
        default_train=200_000, default_val=2_000),  # 推荐条数（条 = 一段文本样本）
}


def download_corpus(max_train_stories, max_val_stories, dataset):  # 下载指定条数样本，写成训练/验证两份纯文本语料
    """流式下载所选数据集，写出到 data/corpus_train.txt 与 data/corpus_val.txt。"""
    spec = DATASETS[dataset]  # 取出所选数据集的配置
    DATA_DIR.mkdir(exist_ok=True)  # 创建数据目录（已存在则跳过）
    if dataset == "minimind":  # minimind 仓库含多个 jsonl，需指定文件
        ds = load_dataset(spec["hf_id"], data_files=spec["data_file"], split="train", streaming=True)  # 流式读取精简版预训练文件
    else:  # tinystories 及其他单配置数据集
        ds = load_dataset(spec["hf_id"], split="train", streaming=True)  # 流式加载：只拉取需要的条数
    paths = {"val": DATA_DIR / "corpus_val.txt", "train": DATA_DIR / "corpus_train.txt"}  # 两份语料的输出路径（流里先取的划给验证集）
    files = {name: open(path, "w", encoding="utf-8") for name, path in paths.items()}  # 同时打开两个输出文件
    try:  # 用 try/finally 保证文件最终一定被关闭
        for i, item in enumerate(ds.take(max_train_stories + max_val_stories)):  # 从数据流中顺序取出指定总条数
            text = item["story"] if "story" in item else item["text"]  # 取出样本正文（兼容不同字段名）
            if not text or not text.strip():  # 空样本直接跳过
                continue  # 处理下一条
            split = "val" if i < max_val_stories else "train"  # 前面一部分划给验证集，其余划给训练集
            files[split].write(text.strip() + "\n<|endoftext|>\n")  # 写入一条样本，样本之间用 <|endoftext|> 标记边界
            if (i + 1) % 5000 == 0:  # 每下载 5000 条打印一次进度
                print(f"已下载 {i + 1} 条样本")
    finally:  # 无论是否异常都执行收尾
        for f in files.values():  # 遍历两个文件句柄
            f.close()  # 关闭文件，落盘
    return paths["train"], paths["val"]  # 返回两份语料的路径


def train_tokenizer(corpus_path, vocab_size):  # 在训练语料上训练 BPE 分词器
    """训练 ByteLevel BPE 并保存到 data/tokenizer.json，返回训练好的分词器。"""
    tok = Tokenizer(models.BPE())  # 创建空的 BPE 分词器
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)  # 预切分：按 GPT-2 方式把文本拆成"词"并在词内标记空格
    tok.decoder = decoders.ByteLevel()  # 解码器：把 token 序列还原成文本时能正确恢复空格
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,  # 词表大小（含特殊 token）
        special_tokens=["<|endoftext|>"],  # 特殊 token：故事边界符，固定占用词表 id 0
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 保证 256 个字节符号都进词表：任何字符都至少能按字节编码，不会失败
        show_progress=True,  # 训练时显示进度条
    )
    tok.train([str(corpus_path)], trainer)  # 在语料文件上统计子词合并规则
    tok.save(str(DATA_DIR / "tokenizer.json"))  # 保存分词器，供采样脚本加载
    return tok  # 返回训练好的分词器


def encode_to_bin(tok, corpus_path, bin_path):  # 把一份语料编码成 token 并写成二进制文件
    """用分词器把整份语料编码为 uint16 token 序列并落盘，返回 token 总数。"""
    text = corpus_path.read_text(encoding="utf-8")  # 读入整份语料文本
    ids = tok.encode(text).ids  # 一次性编码（<|endoftext|> 会被识别为特殊 token，占一个 id）
    np.array(ids, dtype=np.uint16).tofile(bin_path)  # 写成 uint16 二进制（词表 8k 远小于 65535，uint16 足够）
    return len(ids)  # 返回 token 总数


def main():  # 主流程：下载 → 训分词器 → 生成两个 bin
    parser = argparse.ArgumentParser(description="下载预训练语料并生成训练数据")  # 命令行入口
    parser.add_argument("--dataset", type=str, default="tinystories", choices=list(DATASETS.keys()),  # 语料选择
                        help="预训练数据集：tinystories=英文童话（默认），minimind=中文混合语料（几百 MB 精简版）")  # 说明
    parser.add_argument("--max_train_stories", type=int, default=None, help="训练集条数上限（默认取所选数据集的推荐值）")  # 训练规模
    parser.add_argument("--max_val_stories", type=int, default=None, help="验证集条数上限（默认取所选数据集的推荐值）")  # 验证规模
    parser.add_argument("--vocab_size", type=int, default=8192, help="BPE 词表大小")  # 词表规模
    args = parser.parse_args()  # 解析命令行参数
    spec = DATASETS[args.dataset]  # 所选数据集配置
    max_train = args.max_train_stories if args.max_train_stories is not None else spec["default_train"]  # 训练条数（未指定用推荐值）
    max_val = args.max_val_stories if args.max_val_stories is not None else spec["default_val"]  # 验证条数（未指定用推荐值）

    train_path, val_path = download_corpus(max_train, max_val, args.dataset)  # 第一步：下载数据
    print("开始训练 BPE 分词器 ...")  # 第二步提示
    tok = train_tokenizer(train_path, args.vocab_size)  # 第二步：在训练语料上训分词器
    print(f"分词器词表大小: {tok.get_vocab_size()}")  # 打印实际词表大小

    n_train = encode_to_bin(tok, train_path, DATA_DIR / "train.bin")  # 第三步：编码训练集
    n_val = encode_to_bin(tok, val_path, DATA_DIR / "val.bin")  # 第三步：编码验证集
    print(f"训练集 {n_train:,} tokens -> data/train.bin")  # 打印训练集规模
    print(f"验证集 {n_val:,} tokens -> data/val.bin")  # 打印验证集规模
    print("数据准备完成")  # 结束提示


if __name__ == "__main__":  # 作为脚本直接运行时才执行
    main()  # 调用主流程
