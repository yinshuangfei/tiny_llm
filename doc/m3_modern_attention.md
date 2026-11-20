# M3：现代注意力（RoPE + GQA）

## 目标

把 M1 中的 `nn.MultiheadAttention` 升级为手写因果注意力，并加入两个现代 LLM 常用的机制：

1. **RoPE（Rotary Position Embedding）**
   - 用旋转位置编码替换可学习的绝对位置嵌入。
   - Q/K 在 head 内按固定频率旋转。
   - 位置信息通过 Q/K 的相对角度进入 attention score。

2. **GQA（Grouped-Query Attention）**
   - Q 保留原来的 head 数。
   - K/V 使用更少的 `kv_heads`。
   - 本实现默认取 `kv_heads = max(1, heads // 2)`。
   - 推理时能减少 KV cache 显存/内存占用。

## 实现方式

### RoPE

```python
def apply_rope(x, positions):
    # x: (batch, heads, seq_len, head_dim)
    # positions: (seq_len,)
    ...
```

这里采用常见的 half-rotation 实现：

```text
x1 = x[..., : head_dim // 2]
x2 = x[..., head_dim // 2 :]

out1 = x1 * cos - x2 * sin
out2 = x1 * sin + x2 * cos
```

注意 Q/K 进入 RoPE 前已经是：

```text
(B, H, T, D)
```

不是 `(B, T, H, D)`。

### 因果注意力

使用 PyTorch 2.x 的：

```python
F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

避免手工构造大 mask，同时让实现更接近现代框架路径。

### Block 结构

M3 中 `Block` 支持两种模式：

```text
use_rope=False, use_gqa=False
    -> 保留旧实现，兼容 M1 checkpoint

use_rope=True 或 use_gqa=True
    -> 使用新的 CausalSelfAttention
```

因此旧模型仍然可以加载。

## 训练配置

Makefile 中默认启用：

```bash
make train
```

等价于传入：

```bash
--use-rope --use-gqa
```

默认参数：

```text
tokenizer = BPE
vocab_size = 300
steps = 1000
seq_len = 64
dim = 64
layers = 2
heads = 4
lr = 3e-4
batch_size = 8
```

## 测试

新增：

```text
tests/test_modern_attention.py
```

测试两件事：

1. RoPE + GQA 模型可以正常前向传播。
2. 因果性成立：改变序列最后一个 token，不影响前面位置的 logits。

运行：

```bash
make test
```

当前结果：

```text
10 passed
```

## 实验结果

使用 RoPE + GQA 训练：

```text
step 100  loss 0.5881
step 500  loss 0.0228
step 1000 loss 0.0150
```

采样：

```bash
make sample PROMPT="the tiny model learns to" TOKENS=120 TEMPERATURE=0.2
```

输出能稳定复现训练数据中的短句，例如：

```text
the tiny model learns to predict the next character.
the tiny model learns to repeat the pattern.
```

## 现状与下一步

M3 完成了现代注意力的最小可用版本。

下一步 M4 可以做：

1. 增加真正的 KV cache 推理。
2. 支持 streaming generation。
3. 为 RoPE/GQA 添加更细的 shape 单元测试。
4. 模型规模继续放大。
