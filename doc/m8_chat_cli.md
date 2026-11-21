# M8：模型调用交互入口

这个阶段从 `sample.py` 演化出一个更方便使用的交互入口 `tiny_llm.chat`。它不是训练新的模型能力，而是把已有的采样、KV cache、stop string、temperature/top-p/repetition-penalty 能力包成可对话使用的 API。

## 使用方式

### 1. 一次性生成

```bash
PYTHONPATH=src python -m tiny_llm.chat checkpoints/tiny_bpe.pt \
  --prompt "the tiny model learns to" \
  --tokens 80 \
  --temperature 0.8 \
  --top-p 0.9 \
  --repetition-penalty 1.2 \
  --stop "pattern." \
  --seed 7 \
  --once
```

### 2. 进入交互模式

```bash
PYTHONPATH=src python -m tiny_llm.chat checkpoints/tiny_bpe.pt
```

也可以先给一个开场 prompt，然后继续交互：

```bash
make chat CHAT_PROMPT="the tiny model learns to" CHAT_TOKENS=80
```

交互命令：

```text
/help        查看命令
/settings    查看当前采样参数
/set temperature 0.7
/set top_p 0.9
/set top_k 0
/set repetition_penalty 1.2
/set stop pattern.
/reset       清空对话上下文
/exit        退出
```

### 3. Python API

```python
from tiny_llm.chat import ChatSession, ChatSettings
from tiny_llm.train import load_checkpoint

model, tokenizer = load_checkpoint("checkpoints/tiny_bpe.pt")
session = ChatSession(
    model,
    tokenizer,
    ChatSettings(
        max_new_tokens=80,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.2,
        stop_sequences=["pattern."],
        seed=7,
    ),
)

reply = session.respond("the tiny model learns to")
print(reply)
```

## 设计说明

- `plain` 模式适合当前 base model：生成时只把最新的 user 输入作为 continuation prompt；历史仍保留在 `ChatSession.turns` 中。
- `chat` 模板使用 `System:` / `User:` / `Assistant:` 结构，为后续 SFT 做准备。
- `respond()` 会在每轮后把 assistant 回复写回历史。
- 流式输出只打印新增文本，不重复打印整段 prompt。
- stop string 使用 M5 已有的 cutoff 逻辑，避免跨 token 停止词越界。
- 当前模型没有 instruction tuning，因此它更适合补全已有句式，还不能可靠地执行指令。

## 测试

```bash
PYTHONPATH=src python -m pytest -q
```

新增测试覆盖：

- 对话历史拼接；
- `plain` 与 `chat` 模板；
- reset / settings 命令；
- checkpoint 加载后的一次性生成。
