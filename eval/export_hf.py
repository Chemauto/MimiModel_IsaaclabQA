# -*- coding: utf-8 -*-
"""把 mimi_gpt checkpoint 导出成 HuggingFace 格式，供 lm-eval-harness 等标准工具加载。

导出到指定目录，包含：
  config.json               模型配置（auto_map 指向随目录分发的自定义模型代码）
  modeling_mimi_gpt.py      HF 兼容的模型实现（PreTrainedModel 子类，权重名与 model.py 完全一致）
  model.safetensors         权重
  tokenizer.json 等         分词器

用法（在项目根目录执行）：
  python eval/export_hf.py --ckpt out/model_sft.pt --out_dir out_hf/mimi_sft
"""

import argparse  # 命令行参数解析库
import importlib.util  # 动态导入导出目录里的模型代码
import os  # 路径操作
import sys  # 系统路径

import torch  # 权重加载
from tokenizers import Tokenizer  # 原始分词器
from transformers import PreTrainedTokenizerFast  # HF 分词器包装

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录 mimi_gpt/
sys.path.insert(0, ROOT_DIR)  # 导入根目录的 model.py
from model import GPT, GPTConfig  # 我们自己的模型定义

# HF 兼容模型代码模板：写入导出目录随模型分发。
# 模块结构与参数名和 model.py 完全一致，state_dict 可直接迁移；线性层沿用 matmul 实现，推理无 dropout。
MODELING_CODE = '''# -*- coding: utf-8 -*-
"""mimi_gpt 的 HuggingFace 兼容模型实现（由 eval/export_hf.py 生成）。"""
import math
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


class MimiGPTConfig(PretrainedConfig):
    model_type = "mimi_gpt"

    def __init__(self, n_layer=6, n_head=6, n_embd=384, block_size=256, vocab_size=8192, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.dropout = dropout
        # transformers 内部（save/generation 等）会读取标准属性名，这里提供别名
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.hidden_size = n_embd
        self.max_position_embeddings = block_size


class MatmulLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        out = x @ self.weight.t()
        if self.bias is not None:
            out = out + self.bias
        return out


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.qkv = MatmulLinear(config.n_embd, 3 * config.n_embd)
        self.proj = MatmulLinear(config.n_embd, config.n_embd)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=x.dtype))
        att = torch.softmax(att.masked_fill(mask == 0, float("-inf")), dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = MatmulLinear(config.n_embd, 4 * config.n_embd)
        self.proj = MatmulLinear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.proj(torch.nn.functional.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = Mlp(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MimiGPTForCausalLM(PreTrainedModel):
    config_class = MimiGPTConfig
    main_input_name = "input_ids"

    def __init__(self, config):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = MatmulLinear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = nn.Parameter(torch.empty(config.vocab_size, config.n_embd))  # 导出版为独立副本（数值与训练时的绑定权重一致），便于 safetensors 保存
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, MatmulLinear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids=None, attention_mask=None, labels=None, return_dict=True, **kwargs):
        input_ids = input_ids[:, -self.config.block_size:]  # 超长时保留最近的上下文
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.wte(input_ids) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if labels is not None:  # HF 约定：labels 就是输入右移前的完整序列，内部负责移位
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
        if not return_dict:
            return (logits, loss)
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}  # 无 KV 缓存：每步用完整序列重新前向（模型很小，速度足够）
'''


def main():  # 主流程：读 checkpoint → 写模型代码 → 迁移权重 → 保存分词器 → 自验加载
    parser = argparse.ArgumentParser(description="导出 mimi_gpt 为 HuggingFace 格式")  # 命令行入口
    parser.add_argument("--ckpt", type=str, default=os.path.join(ROOT_DIR, "out", "model_sft.pt"), help="要导出的 checkpoint")  # 来源
    parser.add_argument("--out_dir", type=str, default=os.path.join(ROOT_DIR, "out_hf", "mimi_sft"), help="导出目录")  # 目标
    parser.add_argument("--tokenizer", type=str, default=os.path.join(ROOT_DIR, "data", "tokenizer.json"), help="分词器路径")  # 分词器
    args = parser.parse_args()  # 解析

    os.makedirs(args.out_dir, exist_ok=True)  # 创建导出目录
    modeling_path = os.path.join(args.out_dir, "modeling_mimi_gpt.py")  # 模型代码落点
    with open(modeling_path, "w", encoding="utf-8") as f:  # 写出 HF 兼容模型代码
        f.write(MODELING_CODE)  # 模板即代码

    ckpt = torch.load(args.ckpt, map_location="cpu")  # 读取 checkpoint
    src = GPT(GPTConfig(**ckpt["config"]))  # 按配置重建我们自己的模型
    src.load_state_dict(ckpt["model"])  # 载入权重
    src.eval()  # 推理模式

    spec = importlib.util.spec_from_file_location("modeling_mimi_gpt", modeling_path)  # 动态导入刚写出的模块
    module = importlib.util.module_from_spec(spec)  # 构建模块对象
    spec.loader.exec_module(module)  # 执行模块代码

    c = ckpt["config"]  # 配置简写
    hf_config = module.MimiGPTConfig(n_layer=c["n_layer"], n_head=c["n_head"], n_embd=c["n_embd"],  # 按 checkpoint 配置建 HF 配置
                                     block_size=c["block_size"], vocab_size=c["vocab_size"], dropout=0.0)  # 推理无 dropout
    hf_config.architectures = ["MimiGPTForCausalLM"]  # 声明架构名
    hf_config.tie_word_embeddings = False  # 导出为独立副本（数值与绑定版完全一致），规避 transformers 对自定义类绑定权重的保存问题
    hf_config.auto_map = {"AutoConfig": "modeling_mimi_gpt.MimiGPTConfig",  # trust_remote_code 加载时使用的类
                          "AutoModelForCausalLM": "modeling_mimi_gpt.MimiGPTForCausalLM"}  # 同上

    hf_model = module.MimiGPTForCausalLM(hf_config)  # 实例化 HF 模型
    sd = {k: (v.clone() if k == "head.weight" else v) for k, v in src.state_dict().items()  # head.weight 克隆成独立存储
          if not k.endswith("attn.mask")}  # 过滤因果掩码 buffer（HF 实现是动态生成的，不占权重）
    hf_model.load_state_dict(sd, strict=True)  # 权重名一致，可直接严格迁移
    hf_model.save_pretrained(args.out_dir, safe_serialization=True)  # 写 config.json + model.safetensors

    tok = Tokenizer.from_file(args.tokenizer)  # 加载原始 BPE 分词器
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok,  # 包装成 HF 分词器
                                     bos_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|endoftext|>")  # 特殊 token
    hf_tok.save_pretrained(args.out_dir)  # 写 tokenizer_config.json + tokenizer.json
    print(f"已导出 HF 模型到 {args.out_dir}")  # 完成提示

    from transformers import AutoModelForCausalLM, AutoTokenizer  # 自验：按标准方式重新加载
    m = AutoModelForCausalLM.from_pretrained(args.out_dir, trust_remote_code=True)  # 加载模型
    t = AutoTokenizer.from_pretrained(args.out_dir, trust_remote_code=True)  # 加载分词器
    m.eval()  # 评估模式
    prompt = "Instructions: Write a short story about a cat.\nStory:"  # 测试提示
    inputs = t(prompt, return_tensors="pt")  # 编码
    out = m.generate(**inputs, max_new_tokens=30, do_sample=True, top_k=50, temperature=0.8)  # 生成
    print("自验生成：", t.decode(out[0], skip_special_tokens=True)[:150])  # 打印生成片段


if __name__ == "__main__":  # 作为脚本直接运行时才执行
    main()  # 调用主流程
