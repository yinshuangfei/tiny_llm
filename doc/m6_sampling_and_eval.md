# M6：采样控制与评估基础

## 目标

在 M5 的流式生成之上，补齐小模型常用的推理控制，并建立最小 perplexity 评估入口。

## 采样控制

`tiny_llm.sample` 新增参数：

```text
--temperature            温度；0 表示 greedy
--top-k                  只在概率最高的 k 个 token 中采样
--top-p                  nucleus sampling，保留累计概率 >= p 的最小集合
--repetition-penalty     对已出现 token 的 logits 做 CTRL-style penalty
--stop                   stop string，可多次传入
--seed                   固定随机种子，便于复现实验
```

Makefile 示例：

```bash
make sample \
  PROMPT="the tiny model learns to" \
  TOKENS=80 TEMPERATURE=0.8 \
  TOP_P=0.9 REPETITION_PENALTY=1.2 \
  STOP="pattern." SEED=7
```

等价于：

```bash
python -m tiny_llm.sample checkpoints/tiny_bpe.pt \
  --prompt "the tiny model learns to" \
  --tokens 80 --temperature 0.8 \
  --top-p 0.9 --repetition-penalty 1.2 \
  --stop "pattern." --seed 7 --stream
```

### repetition penalty

对已经出现在上下文中的 token：

```text
logit > 0: logit / penalty
logit < 0: logit * penalty
```

`penalty=1.0` 表示不惩罚。

### top-p

按概率从高到低排序，保留累计概率达到 `top_p` 的最小集合；第一个超过阈值的 token 会保留。`top_p=1.0` 表示关闭。

### stop string

CLI 在解码后的文本上检查 stop string。为了处理跨 token 的 stop sequence，流式输出会保守地缓冲尾部字符；确认不会再成为 stop 前缀后才打印。

## Perplexity 评估

新增：

```bash
make eval DATA=data/tiny.txt BATCH_SIZE=8
```

等价于：

```bash
PYTHONPATH=src python -m tiny_llm.eval \
  checkpoints/tiny_bpe.pt data/tiny.txt --batch-size 8
```

评估方式：

1. 用 checkpoint 自带的 tokenizer 编码文本。
2. 切成非重叠的 `max_seq_len + 1` 窗口。
3. 前面 `max_seq_len` 个 token 作为输入，最后一个 token 作为 target。
4. 对所有 token 求 cross entropy 平均。
5. 计算 `perplexity = exp(loss)`。

当前 tiny 语料结果：

```text
loss=0.0069
perplexity=1.0069
tokens=8000
windows=125
```

这个值很高说明模型对这个小型重复语料过拟合，是预期现象；它主要用于验证训练和评估链路。

## 测试

新增：

```text
tests/test_sampling.py
tests/test_eval.py
```

覆盖：

1. `temperature=0` 走 greedy。
2. top-p 能过滤低概率 token。
3. repetition penalty 会改变已重复 token 的选择。
4. stop token 不会出现在生成结果中。
5. CLI 参数解析正确。
6. eval 返回有限 loss / perplexity。

当前结果：

```text
21 passed
```

## 下一步

M7 可以进入训练系统升级：warmup + cosine scheduler、checkpoint resume、更完整的 logging，以及可配置训练配置文件。
