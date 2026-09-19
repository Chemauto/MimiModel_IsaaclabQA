# -*- coding: utf-8 -*-
"""极简 GPT 模型定义（参考 nanoGPT，只保留核心组件）。

结构：token 嵌入 + 位置嵌入 → N × (因果自注意力 + MLP，均为预归一化 + 残差)
→ 最终 LayerNorm → 输出投影（与词嵌入共享权重）。
"""

import math  # 数学库，用于计算初始化标准差
from dataclasses import dataclass  # 数据类装饰器，用于定义配置

import torch  # PyTorch 核心库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 函数式接口：注意力、GELU、交叉熵等


@dataclass
class GPTConfig:  # 模型超参数配置
    n_layer: int = 6  # Transformer 块数
    n_head: int = 6  # 注意力头数
    n_embd: int = 384  # 模型维度（词向量长度）
    block_size: int = 256  # 上下文长度（一次最多看多少个 token）
    vocab_size: int = 8192  # 词表大小
    dropout: float = 0.1  # dropout 比例


class CausalSelfAttention(nn.Module):  # 因果多头自注意力（只能看前文，不能看未来）
    def __init__(self, config):  # 传入模型配置
        super().__init__()  # 调用父类构造函数
        assert config.n_embd % config.n_head == 0, "n_embd必须能被n_head整除"  # 保证每个头的维度是整数
        self.n_head = config.n_head  # 保存头数
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)  # 一个线性层同时算出 Q/K/V 三份投影
        self.proj = nn.Linear(config.n_embd, config.n_embd)  # 注意力输出投影
        self.attn_dropout = config.dropout  # 注意力权重上的 dropout 概率（传给 SDPA 内核）
        self.resid_dropout = nn.Dropout(config.dropout)  # 输出投影后的 dropout

    def forward(self, x):  # x: (batch, seq_len, n_embd)
        B, T, C = x.shape  # 批大小、序列长度、模型维度
        q, k, v = self.qkv(x).split(C, dim=2)  # 投影后切成三份，各自 (B, T, C)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # Q 重排为多头 (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # K 同上
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # V 同上
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout if self.training else 0.0, is_causal=True)  # PyTorch 融合注意力内核：因果掩码 + 缩放点积（自动用 Flash Attention）
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 合并多头回 (B, T, C)
        return self.resid_dropout(self.proj(y))  # 输出投影 + dropout


class MLP(nn.Module):  # 位置前馈网络
    def __init__(self, config):  # 传入模型配置
        super().__init__()  # 调用父类构造函数
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)  # 第一层：升维 4 倍
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)  # 第二层：降回原维度
        self.dropout = nn.Dropout(config.dropout)  # 输出 dropout

    def forward(self, x):  # x: (batch, seq_len, n_embd)
        return self.dropout(self.proj(F.gelu(self.fc(x))))  # fc → GELU 激活 → proj → dropout


class Block(nn.Module):  # 一个 Transformer 块（pre-LN 结构）
    def __init__(self, config):  # 传入模型配置
        super().__init__()  # 调用父类构造函数
        self.ln1 = nn.LayerNorm(config.n_embd)  # 注意力分支前的层归一化
        self.attn = CausalSelfAttention(config)  # 因果自注意力
        self.ln2 = nn.LayerNorm(config.n_embd)  # MLP 分支前的层归一化
        self.mlp = MLP(config)  # 前馈网络

    def forward(self, x):  # x: (batch, seq_len, n_embd)
        x = x + self.attn(self.ln1(x))  # 残差连接：注意力分支
        x = x + self.mlp(self.ln2(x))  # 残差连接：MLP 分支
        return x  # 输出形状与输入相同


class GPT(nn.Module):  # 完整 GPT 模型
    def __init__(self, config):  # 传入 GPTConfig
        super().__init__()  # 调用父类构造函数
        self.config = config  # 保存配置（保存/恢复 checkpoint 时要用）
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token 嵌入表：词 id → 向量
        self.wpe = nn.Embedding(config.block_size, config.n_embd)  # 位置嵌入表：位置 → 向量
        self.drop = nn.Dropout(config.dropout)  # 嵌入后的 dropout
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])  # 堆叠 N 个 Transformer 块
        self.ln_f = nn.LayerNorm(config.n_embd)  # 最后的层归一化
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 输出投影到词表
        self.head.weight = self.wte.weight  # 权重绑定：输入嵌入与输出投影共享同一矩阵，省参数且小模型上更稳
        self.apply(self._init_weights)  # 递归初始化所有子模块
        for pn, p in self.named_parameters():  # 遍历所有参数
            if pn.endswith("proj.weight"):  # 只处理残差分支的输出投影
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))  # 按深度缩小初始化方差，稳定深层训练（nanoGPT 技巧）

    def _init_weights(self, module):  # 默认初始化规则
        if isinstance(module, nn.Linear):  # 线性层
            nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 权重用 N(0, 0.02) 正态初始化
            if module.bias is not None:  # 若有偏置
                nn.init.zeros_(module.bias)  # 偏置置零
        elif isinstance(module, nn.Embedding):  # 嵌入层
            nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 同样用 N(0, 0.02) 初始化

    def forward(self, idx, targets=None):  # idx: (B, T) 输入 token id；targets: (B, T) 每个位置的下一个 token
        B, T = idx.shape  # 取批大小和序列长度
        assert T <= self.config.block_size, f"序列长度 {T} 超过 block_size {self.config.block_size}"  # 位置嵌入表有限，不能超长
        pos = torch.arange(T, device=idx.device)  # 位置索引 0..T-1
        x = self.drop(self.wte(idx) + self.wpe(pos))  # token 嵌入 + 位置嵌入，再 dropout
        for block in self.blocks:  # 依次通过每个 Transformer 块
            x = block(x)  # 逐层更新表示
        x = self.ln_f(x)  # 最终层归一化
        logits = self.head(x)  # 投影到词表：(B, T, vocab_size) 的预测分布
        loss = None  # 不给目标时不算损失（采样阶段用）
        if targets is not None:  # 训练时给定了目标
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))  # 对所有位置求平均交叉熵
        return logits, loss  # 返回预测和损失

    @torch.no_grad()  # 采样不需要梯度
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):  # 从 idx 开始续写指定数量的 token
        for _ in range(max_new_tokens):  # 逐 token 自回归生成
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]  # 超出上下文窗口时只保留最近 block_size 个 token
            logits, _ = self(idx_cond)  # 前向，只取 logits
            logits = logits[:, -1, :] / temperature  # 取最后一个位置的分布并除以温度（<1 更保守，>1 更发散）
            if top_k is not None:  # top-k 截断：只保留概率最高的 k 个 token
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # 找出第 k 大的分数作为阈值
                logits[logits < v[:, [-1]]] = -float("inf")  # 低于阈值的置为 -inf，softmax 后概率为 0
            probs = F.softmax(logits, dim=-1)  # 归一化成概率分布
            idx_next = torch.multinomial(probs, num_samples=1)  # 按概率随机采样下一个 token
            idx = torch.cat((idx, idx_next), dim=1)  # 拼到序列末尾
            if eos_id is not None and idx_next.item() == eos_id:  # 采样到结束符则提前停止
                break  # 退出生成循环
        return idx  # 返回完整序列（含开头）
