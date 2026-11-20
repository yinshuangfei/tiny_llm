# M4：KV Cache 与流式生成

## 目标

M3 完成现代注意力后，生成阶段仍然是每生成一个 token 就重新前向整个上下文：

```text
输入长度 N，生成 T 个 token
时间复杂度约为 O((N + T) ^ 2)
```

M4 引入 KV Cache：

```text
prefill 阶段：一次性处理 prompt，缓存每层 K/V
decode 阶段：每步只计算新 token 的 Q/K/V
时间复杂度明显降低
```

## 设计

### Cache 结构

每层注意力维护一个独立 cache：

```python
kv_cache = [
    {"k": tensor, "v": tensor},  # layer 0
    {"k": tensor, "v": tensor},  # layer 1
    ...
]
```

K/V 的形状是：

```text
(B, kv_heads, cached_len, head_dim)
```

注意这里在 GQA `repeat_interleave` 之前缓存，因此 cache 只保存真实的 KV head 数，不保存扩展后的 Q head 数。

### Forward 接口

```python
logits = model(input_ids, kv_cache=cache)
```

- 第一次传入完整 prompt，创建 prefill cache。
- 后续每次只传入一个新 token。
- 对旧模型路径仍然保持兼容：`model(input_ids)` 不变。

### RoPE 位置

使用 cache 时，新 token 的 position 不是从 0 开始，而是：

```python
position = cached_len + current_index
```

这样 RoPE 的相对位置关系和完整前向一致。

### 因果 mask

三种情况分开处理：

1. **prefill，无 cache**

   ```python
   F.scaled_dot_product_attention(q, k, v, is_causal=True)
   ```

2. **decode，q_len == 1**

   当前 token 可以看到所有历史 token：

   ```python
   F.scaled_dot_product_attention(q, k, v)
   ```

3. **chunked prefill，有 cache 且 q_len > 1**

   构造带偏移的因果 mask。

## 生成接口

模型新增：

```python
model.generate(
    input_ids,
    max_new_tokens,
    temperature,
    top_k,
)
```

采样器 `sample.py` 改为调用这个统一入口。

当上下文超过 `max_seq_len` 时，采用 sliding-window fallback：重新用最近 `max_seq_len` 个 token prefill。

## 测试

新增 `tests/test_kv_cache.py`：

1. cached forward 和 full forward 的 logits 必须一致。
2. cached generation 能正常生成指定长度。

调试过程中发现一个关键问题：cache 列表初始化为 `[None, None]` 后，必须在 forward 中为每层真正创建 dict。否则 prefill 后 cache 仍为空，decode 阶段会丢失上下文。

测试结果：

```text
12 passed
```

## 效果

继续使用 M3 训练出的 checkpoint：

```bash
make sample PROMPT="the tiny model learns to" TOKENS=120 TEMPERATURE=0.2
```

输出仍然正常：

```text
the tiny model learns to predict the next character.
the tiny model learns to repeat the pattern.
```

M4 完成后，生成阶段的重复计算减少，为后续更大上下文和更长时间生成打基础。
