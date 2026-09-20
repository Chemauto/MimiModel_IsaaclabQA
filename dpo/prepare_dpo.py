# -*- coding: utf-8 -*-
"""DPO 偏好数据构造：用 SFT 模型对每条指令采样多个故事，按质量分挑出偏好对。

流程：
  1. 从 TinyStories-instruct 流式取若干条指令（Summary 行）；
  2. SFT 模型对每条指令采样 K 个故事；
  3. 用 sentence-transformers 计算「指令 ↔ 故事」余弦相似度作为质量分；
  4. 每条指令取最高分为 chosen、最低分为 rejected（分差过小则丢弃）；
  5. 打包成 DPO 训练用的二进制文件。

运行后在 data/ 目录产出（不影响其他阶段的产物）：
  dpo_chosen_ids.bin / dpo_chosen_mask.bin       chosen 序列与回答区掩码，各 (n, max_len) 的 int16
  dpo_rejected_ids.bin / dpo_rejected_mask.bin   rejected 序列与回答区掩码，同上
"""

import argparse  # 命令行参数解析库
from pathlib import Path  # 路径操作库
import sys  # 系统路径操作，用于复用 sft/ 目录里的解析函数

import numpy as np  # 写二进制文件
import torch  # 模型采样
from datasets import load_dataset  # 流式下载指令数据
from tokenizers import Tokenizer  # 分词器

ROOT_DIR = Path(__file__).resolve().parent.parent  # 项目根目录 mimi_gpt/（本脚本位于 dpo/ 子目录）
sys.path.insert(0, str(ROOT_DIR / "sft"))  # 复用 sft 目录里的行解析与指令挑选函数
sys.path.insert(0, str(ROOT_DIR))  # 复用根目录的 model.py
from prepare_sft import iter_examples, pick_instruction  # 行块解析与紧凑指令提取
from model import GPT, GPTConfig  # SFT 模型定义

DATA_DIR = ROOT_DIR / "data"  # 数据目录（与各阶段共享，已被 .gitignore 忽略）


@torch.no_grad()  # 采样不需要梯度
def sample_stories(model, tok, instructions, k, device, temperature, top_k):  # 对一批指令各采样 k 个故事
    """返回 stories[i][j] = 第 i 条指令的第 j 个故事文本。"""
    model.eval()  # 评估模式（关闭 dropout）
    stories = [[] for _ in instructions]  # 二维结果容器
    eot_id = tok.token_to_id("<|endoftext|>")  # 结束符 id
    block = model.config.block_size  # 上下文长度
    for i, instr in enumerate(instructions):  # 逐条指令（batch=1，简单清晰；GPU 上单条也很快）
        prompt = f"Instructions: {instr}\nStory:"  # 与 SFT 相同的模板
        start_ids = tok.encode(prompt).ids  # 编码提示
        x = torch.tensor([start_ids], dtype=torch.long, device=device)  # (1, T)
        budget = max(block - len(start_ids), 1)  # 窗口内还能生成的 token 数
        for j in range(k):  # 同一提示采样 k 次
            y = model.generate(x, budget, temperature=temperature, top_k=top_k, eos_id=eot_id)  # 自回归生成
            gen = y[0][len(start_ids):].tolist()  # 只取新生成部分
            text = tok.decode(gen).split("<|endoftext|>")[0].strip()  # 截掉结束符后内容并去空白
            stories[i].append(text)  # 保存
        if (i + 1) % 50 == 0:  # 进度
            print(f"已采样 {i + 1}/{len(instructions)} 条指令")
    return stories  # 返回全部采样结果


def build_rows(tok, prompt_ids, story, max_len):  # 把一条 (提示, 故事) 编码成 (ids, 回答掩码)
    """返回 (ids, mask)；提示过长返回 None。mask=1 表示该位置是回答（参与 DPO 的序列对数似然）。"""
    resp_ids = tok.encode(" " + story).ids  # 故事正文编码
    resp_ids = resp_ids + [tok.token_to_id("<|endoftext|>")]  # 末尾补结束符
    if len(prompt_ids) + len(resp_ids) > max_len:  # 超长截断故事部分（保留开头）
        resp_ids = resp_ids[:max_len - len(prompt_ids)]  # 截断
    if len(resp_ids) < 2:  # 故事太短没有训练价值
        return None  # 跳过
    ids = prompt_ids + resp_ids  # 完整序列
    mask = [0] * len(prompt_ids) + [1] * len(resp_ids)  # 回答区掩码（含结束符）
    pad = max_len - len(ids)  # 补齐长度
    ids = ids + [0] * pad  # 输入补 <|endoftext|>
    mask = mask + [0] * pad  # 掩码补 0
    return ids, mask  # 返回行数据


def main():  # 主流程：取指令 → 采样 → 打分 → 构造偏好对 → 落盘
    parser = argparse.ArgumentParser(description="构造 DPO 偏好数据")  # 命令行入口
    parser.add_argument("--model", type=str, default=str(ROOT_DIR / "out" / "model_sft.pt"), help="用于采样的 SFT 模型（只读）")  # 采样模型
    parser.add_argument("--n_instructions", type=int, default=2000, help="取多少条指令构造偏好对")  # 指令规模
    parser.add_argument("--k", type=int, default=4, help="每条指令采样几个故事")  # 采样宽度
    parser.add_argument("--temperature", type=float, default=0.9, help="采样温度（稍高以保证多样性）")  # 多样性
    parser.add_argument("--top_k", type=int, default=50, help="top-k 截断")  # 采样截断
    parser.add_argument("--min_gap", type=float, default=0.05, help="chosen 与 rejected 的最小分差（过小则丢弃）")  # 偏对质量门槛
    parser.add_argument("--max_len", type=int, default=256, help="单条序列最大 token 数（须不超过模型 block_size）")  # 序列长度
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="打分用句向量模型（英文）")  # 打分模型
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="采样设备")  # 采样设备
    args = parser.parse_args()  # 解析参数

    tok = Tokenizer.from_file(str(DATA_DIR / "tokenizer.json"))  # 与预训练/SFT 一致的分词器
    ckpt = torch.load(args.model, map_location="cpu")  # 读 SFT checkpoint
    model = GPT(GPTConfig(**ckpt["config"])).to(args.device)  # 重建模型
    model.load_state_dict(ckpt["model"])  # 载入权重
    print(f"已加载采样模型 {args.model}（val loss {ckpt['val_loss']:.4f}）")  # 打印信息

    ds = load_dataset("roneneldan/TinyStories-instruct", split="train", streaming=True)  # 指令来源（与 SFT 同一数据集）
    instructions = []  # 收集紧凑指令（Summary 行）
    for instr_text, _story in iter_examples(ds):  # 复用 SFT 的行块解析
        instructions.append(pick_instruction(instr_text))  # 取摘要行作为指令
        if len(instructions) >= args.n_instructions:  # 收满即停
            break  # 退出
    print(f"已收集 {len(instructions)} 条指令，开始采样（每条 {args.k} 个故事）...")  # 进度

    stories = sample_stories(model, tok, instructions, args.k, args.device, args.temperature, args.top_k)  # 采样

    from sentence_transformers import SentenceTransformer, util  # 打分依赖（远程需 pip install sentence-transformers）
    st = SentenceTransformer(args.embed_model)  # 加载句向量模型
    flat = [s for group in stories for s in group if s]  # 展平所有非空故事
    instr_emb = st.encode(instructions, convert_to_tensor=True, show_progress_bar=True)  # 指令向量
    story_emb = st.encode(flat, convert_to_tensor=True, show_progress_bar=True)  # 故事向量
    sims = util.cos_sim(instr_emb, story_emb)  # 相似度矩阵 (n_instr, n_instr*k)

    offset = 0  # 展平前的游标
    pairs = []  # 收集偏好对 (指令, chosen, rejected)
    for i, group in enumerate(stories):  # 逐条指令
        valid = [(s, float(sims[i, offset + j])) for j, s in enumerate(group) if s]  # (故事, 分数)
        offset += len(group)  # 推进游标
        if len(valid) < 2:  # 有效故事不足两个构不成对
            continue  # 跳过
        valid.sort(key=lambda t: t[1])  # 按分数升序
        r_score, chosen_score = valid[0][1], valid[-1][1]  # 最低分与最高分
        if chosen_score - r_score < args.min_gap:  # 分差太小说明模型输出稳定，构不成有意义的偏好
            continue  # 跳过
        pairs.append((instructions[i], valid[-1][0], valid[0][0]))  # (指令, chosen, rejected)
    print(f"构造偏好对 {len(pairs)} 条（指令 {len(instructions)} 条，跳过分差不足者）")  # 汇总

    block = ckpt["config"]["block_size"]  # 与模型上下文一致
    chosen_rows, rejected_rows = [], []  # 收集行数据
    for instr, chosen, rejected in pairs:  # 逐对编码
        prompt_ids = tok.encode(f"Instructions: {instr}\nStory:").ids  # 共享提示
        c = build_rows(tok, prompt_ids, chosen, block)  # chosen 行
        r = build_rows(tok, prompt_ids, rejected, block)  # rejected 行
        if c is None or r is None:  # 编码失败
            continue  # 跳过
        chosen_rows.append(c)  # 收集
        rejected_rows.append(r)  # 收集

    def write(rows, name):  # 把行列表写成 ids 与 mask 两个二进制文件
        ids = np.array([r[0] for r in rows], dtype=np.int16)  # (n, block)
        mask = np.array([r[1] for r in rows], dtype=np.int16)  # (n, block)
        ids.tofile(DATA_DIR / f"dpo_{name}_ids.bin")  # 落盘
        mask.tofile(DATA_DIR / f"dpo_{name}_mask.bin")  # 落盘
        return len(rows)  # 返回行数

    n_c = write(chosen_rows, "chosen")  # 写 chosen
    n_r = write(rejected_rows, "rejected")  # 写 rejected
    print(f"DPO 数据准备完成：{n_c} 对（chosen 与 rejected 行数应一致）→ data/dpo_*.bin")  # 汇总


if __name__ == "__main__":  # 作为脚本直接运行时才执行
    main()  # 调用主流程
