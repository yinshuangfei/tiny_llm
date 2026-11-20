# M5：Streaming Generation CLI

## 目标

M4 已经有 KV cache，但命令行采样仍然等全部 token 生成完才一次性返回。M5 把生成过程改成可以逐 token 输出，并显示生成速度。

## API

模型层新增：

```python
def generate_stream(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> Iterator[int]:
    ...
```

它会逐个 yield 新 token id。

聚合接口保持兼容：

```python
def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    ...
```

`generate()` 现在内部复用 `generate_stream()`，避免两套采样逻辑漂移。

## CLI

新增参数：

```bash
--stream
```

Makefile 默认开启：

```bash
STREAM ?= 1
```

所以下面命令是流式输出：

```bash
make sample TOKENS=60 TEMPERATURE=0.2
```

如果想保持原来的一次性输出：

```bash
make sample STREAM=0 TOKENS=60 TEMPERATURE=0.2
```

流式输出把生成文本写到 stdout，把统计信息写到 stderr：

```text
[generated 60 tokens in 0.064s, 936.5 tokens/s]
```

这样文本仍然可以直接重定向到文件，速度信息不会混进结果。

## 测试

新增：

```text
tests/test_sample_cli.py
tests/test_kv_cache.py 中的 streaming 测试
```

覆盖内容：

1. `--stream` 参数可以被 parser 正确解析。
2. cached 模型下，`generate_stream()` 与 `generate()` 输出一致。
3. 旧 non-cache 模型路径下，流式与非流式输出一致。
4. 流式生成 token 数正确。

当前测试结果：

```text
15 passed
```

## 验证

```bash
make sample TOKENS=60 TEMPERATURE=0.2
```

输出仍能稳定续写训练语料风格：

```text
the tiny model learns to predict the next character.
the tiny model learns to repeat the pattern.
```

## 下一步

M6 可以做更完整的采样控制：

1. top-p / nucleus sampling。
2. repetition penalty。
3. stop token / stop string。
4. 可复现随机种子。
5. 简单 benchmark 工具。
