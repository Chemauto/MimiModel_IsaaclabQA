# 评估 📊

两类评估，分别回答两个问题：**模型在通用基准上处于什么水平**（lm-eval-harness），以及**在本任务上各训练阶段的增益**（自建指标）。

## 一、通用基准（lm-eval-harness）

用业界标准评估工具 [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)（Open LLM Leaderboard 同款）跑四个经典任务：

| 任务 | 考察能力 | 随机猜的水平 |
|---|---|---|
| hellaswag | 常识场景续写 | acc_norm ≈ 25% |
| arc_easy | 小学科学问答 | ≈ 25% |
| piqa | 物理常识 | ≈ 50% |
| winogrande | 指代消解 | ≈ 50% |

**预期管理**：13.89M 参数、2200 万 token 预训练的模型在这些基准上大概率**接近随机水平**——这是正常且有教育意义的：通用能力是规模与数据量的函数，这组数字直观展示了"为什么需要大模型"。

### 步骤

```bash
pip install lm_eval                          # 评估工具
python eval/export_hf.py --ckpt out/model_sft.pt --out_dir out_hf/mimi_sft   # 导出 HF 格式

lm_eval --model hf \
        --model_args pretrained=out_hf/mimi_sft,trust_remote_code=True \
        --tasks hellaswag,arc_easy,piqa,winogrande \
        --device cuda --batch_size 64
```

把 `--ckpt` 换成 `out/model.pt` / `out/model_dpo.pt` 可分别评估预训练版和 DPO 版。

## 二、任务适配评估（自建指标）

对比多个 checkpoint 在「按指令写故事」上的表现：

```bash
python eval/eval_instruction.py --models out/model.pt out/model_sft.pt out/model_dpo.pt --n 50
```

| 指标 | 含义 | 依赖 |
|---|---|---|
| relevance | 指令与生成故事的余弦相似度，越高越贴题 | sentence-transformers |
| distinct-2 | 去重 bigram 占比，越高句式越多样 | 无 |
| completion | 以结束符正常收尾的比例，越高故事越完整 | 无 |
| avg tokens | 平均生成长度 | 无 |

缺 `sentence-transformers` 时脚本自动跳过 relevance 列，其余指标照常。
