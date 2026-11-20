# M1：字符级语言模型

## 目标

建立最小闭环：

```text
text → CharTokenizer → TinyCausalLM → next-token logits → sampling
```

## 当前代码

- `src/tiny_llm/tokenizer.py`：字符级 tokenizer
- `src/tiny_llm/model.py`：decoder-only causal Transformer
- `src/tiny_llm/train.py`：小型训练循环与 checkpoint
- `src/tiny_llm/sample.py`：简单采样脚本
- `tests/`：tokenizer、模型因果性、一步训练与采样测试

## 测试

```bash
make test
```

## 训练与采样

```bash
make train
make sample
make all
```

默认数据文件是 `data/tiny.txt`，默认输出是 `checkpoints/tiny_char.pt`。

可以临时覆盖参数：

```bash
make train STEPS=200 DIM=64 LAYERS=2 HEADS=4
make sample PROMPT=abc TOKENS=100 TEMPERATURE=0.8
```

## 下一步

1. 用小语料跑一次真实训练。
2. 记录 loss 曲线和生成样例。
3. 再进入 BPE tokenizer 与 RoPE/GQA。
