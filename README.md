# tiny_llm

一个从零实现的小型 decoder-only LLM 项目，用于逐步走向现代大语言模型训练与推理系统。

## 当前能力

- char tokenizer / byte-level BPE
- causal Transformer
- RoPE + GQA
- KV cache
- streaming generation
- top-k / top-p / repetition penalty / stop strings
- perplexity eval
- warmup + cosine/linear scheduler
- gradient accumulation
- resumable optimizer checkpoint
- JSON training config
- JSONL training logs

## 常用命令

```bash
make test
make train
make sample PROMPT="the tiny model learns to" TOKENS=80
make eval
make resume RESUME=checkpoints/tiny_bpe.pt STEPS=1200
```

详细设计文档在 `doc/`。
