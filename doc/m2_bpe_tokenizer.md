# M2：BPE Tokenizer

## 目标

把字符级 tokenizer 升级为 byte-level BPE，支持更短的 token 序列和更大的词表。

## 新增文件

- `src/tiny_llm/bpe.py`
- `tests/test_bpe.py`

## 设计

- 基础词表：256 个 byte token。
- 通过 merges 不断合并高频相邻 pair。
- `vocab_size = 256 + len(merges)`。
- 解码时把 token 还原成 bytes，再 decode 为 UTF-8。
- checkpoint 里保存 tokenizer 类型，采样时自动还原 char 或 BPE。

## 训练

```bash
make train
```

默认使用：

```text
TOKENIZER=bpe
VOCAB_SIZE=300
CHECKPOINT=checkpoints/tiny_bpe.pt
```

也可以显式使用字符级：

```bash
make train TOKENIZER=char CHECKPOINT=checkpoints/tiny_char.pt
```

## 采样

```bash
make sample
```

## 当前效果

在 `data/tiny.txt` 上：

```text
chars       = 49000
char_tokens = 49000
bpe_tokens  = 8001
compression = 6.12x
```

## 测试

```bash
make test
```
