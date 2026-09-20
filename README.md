# mimi_gpt 🤖

极简 GPT 实现，用 TinyStories 或 minimind 中文语料训练一个迷你语言模型，完整走一遍「数据准备 → 预训练 → SFT 后训练 → DPO 对齐 → 评估」流程。

## 环境依赖

```bash
pip install torch datasets tokenizers numpy
```

## 三步运行

**1. 准备数据**：自动从 Hugging Face 流式下载所选语料到 `data/` 目录（已写入 `.gitignore`，不会进入 git 记录），然后训练 BPE 分词器、生成训练用的二进制 token 文件。支持 `--dataset` 切换：

```bash
python prepare_data.py --dataset tinystories --max_train_stories 100000 --max_val_stories 2000   # 英文童话（默认），GPU 推荐规模
python prepare_data.py --dataset minimind --max_train_stories 200000 --max_val_stories 2000      # minimind 中文混合语料（pretrain_t2t_mini.jsonl，几百 MB）
python prepare_data.py --dataset minimind --max_train_stories 2000 --max_val_stories 200         # CPU 冒烟验证
```

注意：分词器是在所选语料上现训的，切换数据集后产出的分词器与模型和之前语料的版本**不通用**，需要走完整的预训练 → SFT 流程。

**2. 训练**

```bash
python train.py            # 默认配置，约 10M 参数，GPU 上自动使用 bf16 混合精度（nanoGPT 同款）
python train.py --small    # 冒烟配置，约 1M 参数，CPU 几分钟可见 loss 明显下降
python train.py --dtype fp32    # 强制纯 fp32 训练
```

**3. 续写故事**

```bash
python sample.py --prompt "Once upon a time"
python sample.py --prompt "One day, a little dog" --max_new_tokens 500 --temperature 0.7
```

## 文件说明

| 文件 | 说明 |
|---|---|
| prepare_data.py | 下载预训练语料（TinyStories 英文 / minimind 中文，`--dataset` 切换）、训练 8k 词表 BPE、生成 train.bin / val.bin |
| model.py | 极简 GPT 模型（嵌入 + 因果自注意力 + MLP，权重绑定） |
| train.py | 预训练脚本（下一 token 预测，AdamW + 梯度裁剪） |
| sample.py | 加载 checkpoint，按 temperature + top-k 采样续写 |
| sft/prepare_sft.py | 下载 TinyStories-instruct，打包成「指令 → 故事」样本并生成损失掩码 |
| sft/train_sft.py | SFT 后训练：从 out/model.pt 继续训练，只对故事部分计算损失，保存到 out/model_sft.pt |
| sft/sample_sft.py | 加载 SFT 模型，按指令写故事 |
| dpo/prepare_dpo.py | SFT 模型采样 + 句向量打分，构造「chosen/rejected」偏好对 |
| dpo/train_dpo.py | DPO 对齐训练（policy + 冻结 reference），保存到 out/model_dpo.pt |
| eval/export_hf.py | 把 checkpoint 导出成 HuggingFace 格式 |
| eval/eval_instruction.py | 指令跟随指标（相关性/多样性/完整率）多 checkpoint 对比，详见 [eval/README.md](eval/README.md) |

## SFT 后训练（可选的第二阶段）

在预训练完成后进行，让模型从「自由续写」变成「按指令写故事」。所有 SFT 脚本在 `sft/` 目录下，不修改任何预训练产物：权重只读 `out/model.pt`，结果另存 `out/model_sft.pt`。

```bash
python sft/prepare_sft.py --max_train_samples 50000 --max_val_samples 500   # 下载 instruct 数据并打包（依赖预训练阶段生成的 data/tokenizer.json）
python sft/train_sft.py                                                     # 微调，GPU 上 1000 iter 约几分钟
python sft/sample_sft.py --instructions "Write a short story about a little kitten who makes a new friend."
```

样本模板（提示部分不计算损失，只学习故事生成）：

```text
Instructions: <情节摘要 / 特征 / 词汇要求>
Story: <故事><|endoftext|>
```

## DPO 对齐训练（第三阶段）

用偏好对做直接偏好优化：SFT 模型对同一指令采样多个故事，按「指令-故事」相关性挑出 chosen/rejected，训练让模型偏向好回答。依赖 `pip install sentence-transformers`，且需先完成 SFT（用到 `out/model_sft.pt`）。

```bash
python dpo/prepare_dpo.py --n_instructions 2000    # 采样 + 打分 + 构造偏好对
python dpo/train_dpo.py                            # DPO 训练，保存到 out/model_dpo.pt
python sft/sample_sft.py --model out/model_dpo.pt --instructions "..."   # 用 DPO 模型生成
```

## 评估

通用基准（lm-eval-harness：hellaswag / arc_easy / piqa / winogrande）与任务适配指标（relevance / distinct-2 / completion）的说明与命令见 [eval/README.md](eval/README.md)。

## 超参数

两套预设配置写在 `train.py` 顶部的 `CONFIGS` 字典中，按机器情况修改；`prepare_data.py` 的数据规模、`sample.py` 的采样参数通过命令行参数调整。
