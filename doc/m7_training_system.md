# M7：训练系统升级

## 目标

在最小训练循环上补齐现代训练系统的基础能力：学习率调度、梯度累积、可恢复 checkpoint、训练配置和结构化日志。当前服务器是 CPU 环境，因此暂不引入 mixed precision 和分布式训练。

## 设计

### 1. 学习率调度

新增 `src/tiny_llm/scheduler.py`：

- `constant`：保持峰值学习率；
- `linear`：从峰值线性衰减到 `base_lr * min_lr_ratio`；
- `cosine`：从峰值余弦衰减到 `base_lr * min_lr_ratio`；
- `warmup_steps`：前若干步线性升温；
- optimizer param group 保存 `base_lr`，避免 resume 后丢失原始学习率。

### 2. 梯度累积

`batch_size` 现在表示 micro-batch size：

```text
1 optimizer step = grad_accum 个 micro-batch
```

每个 micro-batch 的 loss 先除以 `grad_accum` 再 backward。多个 micro-batch 的梯度等价取平均，从而在内存受限环境下模拟更大 batch size。

### 3. 可恢复 checkpoint

checkpoint 新增：

- optimizer state；
- 当前完成步数；
- CPU RNG state。

模型、tokenizer 和模型 config 保持兼容。写入时先保存 `.tmp` 文件，再原子替换，减少训练中断造成 checkpoint 损坏的风险。

### 4. JSON 配置

`configs/tiny_bpe.json` 是默认训练配置。CLI 显式参数优先级高于 JSON。

### 5. 日志

可选 `--log-file logs/train.jsonl`，每条记录包含：

- `step`
- `total_steps`
- `loss`
- `lr`
- `tokens`
- `tokens_per_second`
- `step_seconds`

## 用法

```bash
# 按 configs/tiny_bpe.json 训练
make train

# 覆盖超参数
make train STEPS=300 DIM=96 LAYERS=3 WARMUP=50 GRAD_ACCUM=2

# 从 checkpoint 继续训练到 1200 步
make resume RESUME=checkpoints/tiny_bpe.pt STEPS=1200

# 训练并写日志
make train LOG_FILE=logs/train.jsonl
```

## 测试

新增 `tests/test_training_system.py`，覆盖 scheduler、配置覆盖、梯度累积、日志和 resume。运行：

```bash
PYTHONPATH=src python -m pytest -q
```

## 暂缓项

- CPU 暂不启用 BF16/FP16；
- 当前单机 CPU 环境暂不实现 DDP/FSDP；
- 暂用 JSONL，而非 TensorBoard/wandb。
