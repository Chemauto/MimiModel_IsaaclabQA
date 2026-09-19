# mimi_gpt 🤖

极简 GPT 实现，用 TinyStories 数据集训练一个会续写英文小故事的迷你语言模型，完整走一遍「数据准备 → 预训练 → 采样」流程。

## 环境依赖

```bash
pip install torch datasets tokenizers numpy
```

## 三步运行

**1. 准备数据**：自动从 Hugging Face 流式下载 TinyStories 到 `data/` 目录（已写入 `.gitignore`，不会进入 git 记录），然后训练 BPE 分词器、生成训练用的二进制 token 文件。

```bash
python prepare_data.py --max_train_stories 100000 --max_val_stories 2000   # GPU 训练推荐规模
python prepare_data.py --max_train_stories 2000 --max_val_stories 200      # CPU 冒烟验证
```

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
| prepare_data.py | 下载 TinyStories、训练 8k 词表 BPE、生成 train.bin / val.bin |
| model.py | 极简 GPT 模型（嵌入 + 因果自注意力 + MLP，权重绑定） |
| train.py | 预训练脚本（下一 token 预测，AdamW + 梯度裁剪） |
| sample.py | 加载 checkpoint，按 temperature + top-k 采样续写 |

## 超参数

两套预设配置写在 `train.py` 顶部的 `CONFIGS` 字典中，按机器情况修改；`prepare_data.py` 的数据规模、`sample.py` 的采样参数通过命令行参数调整。
