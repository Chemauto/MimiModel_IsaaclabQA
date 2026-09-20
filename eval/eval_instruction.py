# -*- coding: utf-8 -*-
"""任务适配评估：对比多个 checkpoint 在「按指令写故事」上的表现。

指标（对小模型有区分度的三个信号）：
  relevance   指令与生成故事的余弦相似度（sentence-transformers 句向量），越高说明越贴题
  distinct-2  全部生成文本中去重 bigram 占比，越高说明句式越多样（越低越"车轱辘话"）
  completion  以结束符正常收尾的比例，越高说明故事写得完整
  avg tokens  平均生成长度

用法（在项目根目录执行）：
  python eval/eval_instruction.py --models out/model.pt out/model_sft.pt out/model_dpo.pt --n 50
"""

import argparse  # 命令行参数解析库
import os  # 路径操作
import sys  # 系统路径
from collections import OrderedDict  # 有序字典，用于 bigram 统计

import torch  # 模型与生成
from tokenizers import Tokenizer  # 分词器

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录 mimi_gpt/
sys.path.insert(0, ROOT_DIR)  # 导入根目录模块
sys.path.insert(0, os.path.join(ROOT_DIR, "sft"))  # 复用 sft 目录的指令解析
from model import GPT, GPTConfig  # 模型定义
from prepare_sft import iter_examples, pick_instruction  # 从数据流取指令


@torch.no_grad()  # 生成不需要梯度
def generate_stories(model, tok, instructions, device, temperature, top_k, max_new_tokens):  # 对每条指令生成一个故事
    """返回 (故事列表, 完成率)：完成 = 采样到结束符。"""
    model.eval()  # 评估模式
    results, complete = [], 0  # 结果与完成计数
    eot_id = tok.token_to_id("<|endoftext|>")  # 结束符
    block = model.config.block_size  # 上下文长度
    for instr in instructions:  # 逐条生成（清晰优先；大批量可用 batch 优化）
        prompt = f"Instructions: {instr}\nStory:"  # 与 SFT/DPO 训练一致的模板
        start_ids = tok.encode(prompt).ids  # 编码
        x = torch.tensor([start_ids], dtype=torch.long, device=device)  # (1, T)
        budget = min(max_new_tokens, max(block - len(start_ids), 1))  # 窗口内可生成长度
        y = model.generate(x, budget, temperature=temperature, top_k=top_k, eos_id=eot_id)  # 生成
        gen = y[0][len(start_ids):].tolist()  # 取新生成部分
        has_eot = eot_id in gen  # 是否正常收尾
        complete += int(has_eot)  # 统计完成率
        text = tok.decode(gen).split("<|endoftext|>")[0].strip()  # 截掉结束符后内容
        results.append(text)  # 保存
    return results, complete / max(len(instructions), 1)  # 返回故事与完成率


def distinct_n(stories, n=2):  # 语料级 distinct-n：不同 n-gram 占比
    total, uniq = 0, set()  # 总数与去重集合
    for s in stories:  # 遍历每篇故事
        words = s.lower().replace(".", " ").replace(",", " ").replace('"', " ").split()  # 粗分词
        grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]  # n-gram 列表
        total += len(grams)  # 累计总数
        uniq.update(grams)  # 累计去重
    return len(uniq) / max(total, 1)  # 占比


def main():  # 主流程
    parser = argparse.ArgumentParser(description="指令跟随评估：多 checkpoint 对比")  # 命令行入口
    parser.add_argument("--models", type=str, nargs="+", default=[  # 默认对比三个阶段
        os.path.join(ROOT_DIR, "out", "model.pt"),
        os.path.join(ROOT_DIR, "out", "model_sft.pt"),
        os.path.join(ROOT_DIR, "out", "model_dpo.pt")], help="要对比的 checkpoint 列表")
    parser.add_argument("--n", type=int, default=50, help="评估用指令条数")  # 指令数
    parser.add_argument("--max_new_tokens", type=int, default=200, help="每条最多生成 token 数")  # 长度
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度")  # 温度
    parser.add_argument("--top_k", type=int, default=50, help="top-k 截断")  # top-k
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="相关性打分用句向量模型")  # 打分模型
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")  # 设备
    args = parser.parse_args()  # 解析

    from datasets import load_dataset  # 指令来源
    ds = load_dataset("roneneldan/TinyStories-instruct", split="train", streaming=True)  # 流式读取
    instructions = []  # 收集评估指令
    for instr_text, _story in iter_examples(ds):  # 复用 SFT 解析
        instructions.append(pick_instruction(instr_text))  # 取摘要行
        if len(instructions) >= args.n:  # 收满即停
            break  # 退出
    print(f"评估指令 {len(instructions)} 条\n")  # 打印规模

    try:  # 相关性指标依赖 sentence-transformers，缺库时跳过该列
        from sentence_transformers import SentenceTransformer, util  # 导入
        st = SentenceTransformer(args.embed_model)  # 加载句向量模型
        instr_emb = st.encode(instructions, convert_to_tensor=True, show_progress_bar=False)  # 指令向量只算一次
    except Exception as e:  # 缺库或下载失败
        print(f"[提示] 句向量模型不可用（{e.__class__.__name__}），跳过 relevance 指标")  # 提示
        st = None  # 置空

    tok = Tokenizer.from_file(os.path.join(ROOT_DIR, "data", "tokenizer.json"))  # 分词器
    header = f"{'checkpoint':<28}{'relevance':>10}{'distinct-2':>12}{'completion':>12}{'avg tokens':>12}"  # 表头
    print(header)  # 打印表头
    print("-" * len(header))  # 分隔线
    for path in args.models:  # 逐个 checkpoint 评估
        if not os.path.exists(path):  # 文件不存在则跳过
            print(f"{os.path.basename(path):<28}  [跳过：文件不存在]")  # 提示
            continue  # 下一个
        ckpt = torch.load(path, map_location="cpu")  # 读 checkpoint
        model = GPT(GPTConfig(**ckpt["config"])).to(args.device)  # 重建模型
        model.load_state_dict(ckpt["model"])  # 载入权重
        stories, completion = generate_stories(model, tok, instructions, args.device,  # 生成全部故事
                                               args.temperature, args.top_k, args.max_new_tokens)  # 参数
        d2 = distinct_n(stories, 2)  # 多样性
        avg_len = sum(len(s.split()) for s in stories) / max(len(stories), 1)  # 平均词数
        rel_str = "-"  # 相关性默认不展示
        if st is not None:  # 有句向量模型才计算
            story_emb = st.encode(stories, convert_to_tensor=True, show_progress_bar=False)  # 故事向量
            rel = util.cos_sim(instr_emb, story_emb).diagonal().mean().item()  # 对角线：各自指令与自身故事的相关性
            rel_str = f"{rel:.3f}"  # 格式化
        name = os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)  # 展示名
        print(f"{name:<28}{rel_str:>10}{d2:>12.3f}{completion:>12.2%}{avg_len:>12.1f}")  # 打印一行
    print("\n说明：relevance 越高越贴题；distinct-2 越高句式越多样；completion 越高故事越完整；"
          "三个 checkpoint 横向对比即可看出各阶段的增益。")  # 阅读说明


if __name__ == "__main__":  # 作为脚本直接运行时才执行
    main()  # 调用主流程
