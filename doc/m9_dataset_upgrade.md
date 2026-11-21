# M9：语料升级与对话效果优化

## 目标

原来的 `data/tiny.txt` 只有两条句子反复循环，模型很快把训练集背下来，生成结果也几乎总是同一段文本。这次把语料换成更有结构的小型对话语料，让 `chat` 模板能学到 `User:` / `Assistant:` 的模式。

## 语料设计

`data/tiny.txt` 现在包含三类内容：

1. **问答对话**：约 100 组常见问答，覆盖身份、能力、模型概念、简单事实和故事请求。
2. **通用短句**：模型、训练、采样、语言、生活场景相关的多样句子。
3. **故事句式**：固定语法骨架中替换人物、地点、物品，帮助模型学习连贯续写。

问答块被重复多次，这是因为 tiny model 的目标是学习稳定的问答模式，而不是追求开放域泛化。

## 训练配置

这次默认配置从较小模型升级为：

```text
vocab_size = 512
dim = 96
layers = 3
heads = 4
steps = 5000
```

训练命令：

```bash
make train
```

## 效果

在当前训练集上评估：

```text
loss=0.1011
perplexity=1.1064
```

采样示例：

```bash
make chat CHAT_ONCE=1 CHAT_PROMPT="what is your name?"
```

现在能生成类似：

```text
my name is tiny llm.
```

## Chat 默认行为

`make chat` 现在默认使用：

```make
CHAT_TEMPLATE=chat
CHAT_TEMPERATURE=0.2
CHAT_TOP_P=0.95
CHAT_REPETITION_PENALTY=1.15
CHAT_STOP=User:
```

这样输出更稳定，并且会在模型开始编造下一个 `User:` 时停止。

## 边界

这仍然是一个 tiny model，不是 instruction-tuned LLM。它适合演示语料、tokenizer、训练、采样和 chat 接口的关系，但不能替代真实大模型。
