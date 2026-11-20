# tiny_llm：从零到现代 GLM-5.3 风格大语言模型实现计划

> 创建日期：2026-11-17  
> 目标：从零实现一个小型 LLM，逐步演进到现代 GLM-5.3 风格的模型训练、推理与对齐系统。  
> 本计划不承诺复刻任何未公开的模型权重或内部细节，而是按公开、可验证的通用工程路线推进。

## 0. 总体目标

tiny_llm 的发展分为三层：

1. **可用模型**：能完成分词、训练、采样、评测的最小闭环。
2. **现代 LLM**：具备 Transformer Decoder、RoPE、GQA/MQA、FlashAttention、混合精度、分布式训练和高效推理。
3. **GLM-5.3 风格目标**：在公开通用技术基础上，探索长上下文、稀疏 MoE、多阶段对齐、工具调用、推理链和现代 Agent 能力。

最终工程目标不是只训练一个 toy 模型，而是建立一套可持续扩展的完整系统。

---

## 1. 基本原则

### 1.1 从零开始

- 第一阶段避免直接依赖高层框架，例如 DeepSpeed、Megatron、vLLM。
- 先用 Python + PyTorch 手写关键组件。
- 每个里程碑都必须能被最小实验验证。
- 训练日志、数据集、分词器、权重、评估结果全部可追溯。

### 1.2 逐步现代化

不要一开始就实现复杂系统。路线应为：

```text
最小模型 → 可训练 → 可推理 → 可评测 → 可复现 → 可扩展 → 可分布式 → 可对齐 → 可服务 → 可持续演进
```

### 1.3 可评估优先

每个版本都要回答：

- loss 是否下降？
- 困惑度是多少？
- 在什么数据上训练？
- 推理速度是多少？
- 显存占用是多少？
- 与上一版相比改善在哪里？

---

## 2. 项目目录规划

```text
tiny_llm/
├── README.md
├── doc/
│   ├── implementation_plan.md
│   ├── architecture/
│   ├── data/
│   ├── training/
│   ├── inference/
│   ├── alignment/
│   └── experiments/
├── data/
├── tokenizer/
├── model/
├── train/
├── eval/
├── inference/
├── configs/
├── scripts/
├── tests/
└── checkpoints/
```

建议主要模块：

| 模块 | 职责 |
| --- | --- |
| `tokenizer/` | 分词器实现、BPE 训练、vocab 管理 |
| `model/` | embedding、attention、FFN、normalization、Transformer decoder |
| `train/` | dataloader、optimizer、scheduler、checkpoint、日志 |
| `eval/` | perplexity、下游任务、代码/数学/常识评测 |
| `inference/` | 本地推理、KV cache、量化、服务接口 |
| `configs/` | 模型、数据、训练配置 |
| `doc/` | 设计与实验文档 |
| `tests/` | 组件正确性测试 |

---

## 3. 阶段路线图

## M0：环境与工程基线

目标：建立可复现开发环境。

任务：

1. 确认 Python、PyTorch、CUDA 环境。
2. 确认 GPU 型号、显存、驱动和 NCCL 支持。
3. 初始化代码结构、依赖管理、日志、随机种子、配置系统。
4. 添加单元测试框架。
5. 写入最小 README 和项目约定。

验收标准：

- 可以一键运行测试。
- 可以打印 GPU / CUDA / PyTorch 环境信息。
- 代码和文档进入 Git 管理。

---

## M1：字符级小模型先行

目标：用最简单方式跑通“数据 → 模型 → 训练 → 生成”。

任务：

1. 准备小型中文或英文语料。
2. 实现字符级 tokenizer。
3. 实现最简单的 causal language model。
4. 实现 forward、loss、backward、optimizer step。
5. 实现贪心采样、temperature、top-k、top-p。

验收标准：

- 能在 CPU 或单 GPU 上训练。
- loss 明显下降。
- 能生成局部可读文本。
- checkpoint 可以保存和恢复。

交付物：

- `tokenizer/char_tokenizer.py`
- `model/tiny_model.py`
- `train/train_char.py`
- `inference/sample.py`

---

## M2：完整分词系统

目标：从字符 tokenizer 演进到 BPE tokenizer。

任务：

1. 实现基础 Unicode / byte-level BPE。
2. 实现 vocab、merges、special tokens。
3. 实现 encode / decode。
4. 处理中文、英文、数字、空白、代码混合文本。
5. 保存和加载 tokenizer 配置。

验收标准：

- encode → decode 往返无损。
- 可以统计 corpus token 数。
- 支持 special token。
- 分词粒度可控。

交付物：

- `doc/data/tokenizer_design.md`
- `tokenizer/bpe_tokenizer.py`
- `tokenizer/train_bpe.py`

---

## M3：标准 Transformer Decoder

目标：手写一个现代 decoder-only Transformer。

核心组件：

1. Token embedding。
2. Position embedding 或后续替换为 RoPE。
3. Causal multi-head self-attention。
4. Feed-forward network。
5. LayerNorm 或 RMSNorm。
6. Residual connection。
7. LM head。

建议架构：

```text
input_ids
   ↓
TokenEmbedding
   ↓
RMSNorm
   ↓
Self-Attention
   ↓
Residual
   ↓
RMSNorm
   ↓
MLP / SwiGLU
   ↓
Residual
   ↓
Final RMSNorm
   ↓
LM Head
   ↓
next-token logits
```

验收标准：

- causal mask 正确。
- batch、sequence、hidden shape 正确。
- 相同配置下 loss 可稳定下降。
- 参数量、FLOPs、显存占用可统计。

---

## M4：现代注意力机制

目标：逐步升级 attention。

需要实现或验证：

1. RoPE：旋转位置编码。
2. GQA：grouped-query attention。
3. FlashAttention 或等价 efficient attention。
4. KV cache。
5. Sliding window attention，可选。
6. 长上下文外推方案，例如 YaRN / NTK-style scaling，可选。

验收标准：

- RoPE 相对位置语义可验证。
- GQA 减少显存占用且质量不明显下降。
- KV cache 后推理长度增加时速度显著改善。
- 支持不同 context length 配置。

---

## M5：训练系统

目标：建立工业可用的训练循环。

任务：

1. Dataloader：streaming、token packing、sequence length 控制。
2. Optimizer：AdamW 或现代变体。
3. Scheduler：warmup + cosine / linear decay。
4. Gradient clipping。
5. Mixed precision：BF16 / FP16。
6. Gradient accumulation。
7. Checkpoint 保存与恢复。
8. TensorBoard / wandb / 本地日志。
9. 断点续训。
10. 多卡训练，先 DDP，后 FSDP 或 ZeRO。

验收标准：

- 单卡可稳定训练。
- 至少支持 2 GPU。
- 训练中断后可恢复。
- 记录 loss、lr、tokens/sec、GPU memory。

---

## M6：评测系统

目标：建立模型能力观测。

基础评测：

1. perplexity。
2. token accuracy。
3. generation quality。
4. repeated-token and degeneration 检查。

进阶评测：

1. 中文理解。
2. 英文理解。
3. 数学题。
4. 代码生成。
5. 长文档问答。
6. 指令跟随。
7. 工具调用格式正确率。

验收标准：

- 能输出统一 JSON / CSV / Markdown 报告。
- 不同 checkpoint 可以横向对比。
- 训练速度、推理速度、质量均有记录。

---

## M7：后训练与对齐

目标：从 base model 演进为指令模型。

阶段：

1. **SFT**：使用 instruction-response 数据微调。
2. **DPO / RLHF-lite**：使用偏好数据优化回答质量。
3. **Tool SFT**：训练模型理解工具 schema、调用参数和返回结果。
4. **RAG-style reasoning**：训练模型基于检索内容回答。
5. **Agent behavior**：训练多步任务分解、工具使用、错误恢复。

验收标准：

- SFT 后能稳定遵循基础指令。
- 不再只续写文本，而是以 assistant 方式回答。
- 能生成结构化输出。
- 有 rejection / refusal / safety 基线。

---

## M8：推理与服务

目标：模型可以被程序稳定调用。

任务：

1. 本地命令行交互。
2. OpenAI-compatible HTTP API。
3. Streaming response。
4. KV cache。
5. Batch inference。
6. 量化：INT8、INT4、BF16 fallback。
7. 推理框架适配，例如 vLLM 或自定义 runtime。

验收标准：

- 支持单轮、多轮对话。
- 支持 streaming。
- 支持 stop token。
- API 有 schema、日志、错误处理。
- 推理吞吐和延迟可测量。

---

## M9：规模化路线

目标：逐步放大模型规模。

建议从小到大：

```text
nano:    1M  - 10M 参数
tiny:    10M - 100M 参数
small:   100M - 500M 参数
medium:  500M - 1B+ 参数
large:   1B - 10B+ 参数
```

先在小规模验证，再放大。每次放大需要记录：

- hidden size
- layer 数
- head 数
- KV head 数
- FFN ratio
- context length
- vocab size
- token budget
- batch size
- learning rate
- loss curve
- eval score

---

## M10：现代 GLM-5.3 风格目标

现代大型模型通常会组合以下能力。tiny_llm 的远期目标是在公开技术路线下逐步实现类似能力。

### 10.1 架构层

- Decoder-only Transformer。
- RMSNorm。
- SwiGLU 或现代 FFN。
- RoPE / long-context position scaling。
- GQA / MQA。
- FlashAttention。
- 残差与归一化策略。
- 可选：MoE、专家路由、负载均衡损失。
- 可选：linear attention / hybrid attention / sparse attention。

### 10.2 数据层

- 大规模清洗语料。
- 中英混合语料。
- 代码语料。
- 数学与推理语料。
- 高质量指令数据。
- 偏好数据。
- 工具调用数据。
- 长文档数据。
- 多轮对话数据。

数据流程：

```text
raw corpus
  ↓
dedup
  ↓
language id
  ↓
quality filter
  ↓
safety filter
  ↓
domain weighting
  ↓
tokenizer
  ↓
sequence packing
  ↓
training mixture
```

### 10.3 训练层

- BF16 / FP8 混合精度。
- Activation checkpointing。
- Distributed data parallel。
- Tensor parallel。
- Pipeline parallel。
- Sequence parallel。
- FSDP / ZeRO。
- MoE expert parallel。
- 自动容错与断点续训。
- Curriculum training。

### 10.4 对齐层

- Supervised Fine-Tuning。
- Preference optimization。
- Reinforcement learning from human or AI feedback。
- Safety tuning。
- Tool-call tuning。
- Long-context instruction tuning。
- Reasoning-focused RL。
- Agent trajectory training。

### 10.5 能力层

- 多轮对话。
- 长上下文理解。
- 文档级问答。
- 代码理解与生成。
- 数学推理。
- 工具调用。
- 多步任务执行。
- 检索增强。
- 结构化输出。
- 可控生成。

---

## 4. 推荐里程碑顺序

```text
M0 环境准备
 ↓
M1 字符级模型
 ↓
M2 BPE tokenizer
 ↓
M3 Transformer decoder
 ↓
M4 RoPE + GQA + FlashAttention
 ↓
M5 训练系统
 ↓
M6 评测系统
 ↓
M7 SFT / preference tuning
 ↓
M8 推理服务
 ↓
M9 规模化
 ↓
M10 GLM-5.3 风格能力
```

不要跳过 M1-M6。  
如果直接从复杂架构开始，后续很难定位模型、数据、训练或工程哪一层有问题。

---

## 5. 最小实验配置示例

```yaml
model:
  vocab_size: 32000
  hidden_size: 512
  num_layers: 8
  num_attention_heads: 8
  num_key_value_heads: 4
  intermediate_size: 1408
  max_position_embeddings: 2048
  rope_theta: 10000
  norm: rmsnorm
  activation: swiglu
  dropout: 0.0

train:
  seq_len: 1024
  batch_size: 16
  gradient_accumulation_steps: 4
  lr: 3.0e-4
  min_lr: 3.0e-5
  warmup_steps: 200
  weight_decay: 0.1
  max_grad_norm: 1.0
  precision: bf16
  optimizer: adamw

data:
  dataset: tiny_corpus
  tokenizer: bpe
  packing: true

eval:
  eval_every_steps: 500
  save_every_steps: 1000
  metrics:
    - loss
    - perplexity
    - sample_quality
```

---

## 6. 代码组织建议

### tokenizer

```python
class Tokenizer:
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def save(self, path): ...
    @classmethod
    def load(cls, path): ...
```

### model

```python
class TinyLM(nn.Module):
    def forward(self, input_ids, attention_mask=None, past_kv=None):
        ...
        return logits, new_kv
```

### train

```python
for step, batch in enumerate(loader):
    loss = model(**batch).loss
    loss.backward()
    clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
```

---

## 7. 第一周可执行任务

1. 建立 `doc/`、`model/`、`train/`、`tokenizer/`、`configs/`、`tests/`。
2. 写一个最小 Python 训练脚本。
3. 实现字符 tokenizer。
4. 实现一个小 Transformer。
5. 在 1MB 左右文本上训练 500-2000 steps。
6. 生成 3 段文本并记录效果。
7. 写实验报告：`doc/experiments/m1_char_lm.md`。
8. 总结当前 loss、参数量、显存、生成样例和问题。

---

## 8. 风险与对策

| 风险 | 表现 | 对策 |
| --- | --- | --- |
| loss 不下降 | 模型学不到模式 | 检查 mask、optimizer、lr、数据、tokenizer |
| 生成重复 | 采样退化 | temperature、top-k、top-p、repetition penalty |
| 显存不足 | OOM | 减小 batch、seq_len、activation checkpointing、gradient accumulation |
| 训练崩溃 | NaN / inf | 检查 fp16、梯度裁剪、loss normalization、数据异常 |
| 过拟合 | train loss 低但 eval 差 | 增加数据、正则、早停 |
| 分词混乱 | 中文 / 英文 / 代码处理差 | 改用 byte-level BPE，清理语料 |
| 无法分布式 | 多卡失败 | 先单卡稳定，再 DDP，再 FSDP / ZeRO |
| 工程失控 | 功能多但跑不通 | 每个 milestone 都保持最小可运行版本 |

---

## 9. 当前结论

tiny_llm 应先完成一个小而正确的端到端系统，再逐步升级到现代 LLM 架构。

当前最合理的下一步是：

1. 创建基础目录结构。
2. 实现 M1 字符级模型。
3. 用极小语料训练。
4. 保存 checkpoint。
5. 记录第一条实验结果。

