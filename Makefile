PYTHON ?= python
PYTHONPATH ?= src
DATA ?= data/processed/train.txt
TRAIN_DATA ?= data/processed/manifest.json
CHECKPOINT ?= checkpoints/tiny_bpe.pt
CONFIG ?= configs/tiny_bpe.json
TOKENIZER ?= bpe
VOCAB_SIZE ?= 512
STEPS ?= 5000
BATCH_SIZE ?= 8
SEQ_LEN ?= 64
DIM ?= 128
LAYERS ?= 4
HEADS ?= 4
LR ?= 3e-4
WARMUP ?= 50
MIN_LR_RATIO ?= 0.1
SCHEDULER ?= cosine
GRAD_ACCUM ?= 1
LOG_EVERY ?= 50
LOG_FILE ?=
RESUME ?=
PROMPT ?= the tiny model learns to
TOKENS ?= 120
TEMPERATURE ?= 0.2
TOP_K ?= 0
TOP_P ?= 1.0
REPETITION_PENALTY ?= 1.0
STOP ?=
SEED ?= 0
STREAM ?= 1

CHAT_PROMPT ?=
CHAT_SYSTEM ?=
CHAT_TEMPLATE ?= chat
CHAT_TOKENS ?= 80
CHAT_TEMPERATURE ?= 0.2
CHAT_TOP_K ?= 0
CHAT_TOP_P ?= 0.95
CHAT_REPETITION_PENALTY ?= 1.15
CHAT_STOP ?= User:
CHAT_SEED ?=
CHAT_STREAM ?= 1
CHAT_ONCE ?= 0
SHOW_PROMPT ?= 0

ifeq ($(STREAM),1)
SAMPLE_FLAGS := --stream
else
SAMPLE_FLAGS :=
endif
SAMPLE_FLAGS += --seed $(SEED) --top-p $(TOP_P) --repetition-penalty $(REPETITION_PENALTY)
ifneq ($(TOP_K),0)
SAMPLE_FLAGS += --top-k $(TOP_K)
endif
ifneq ($(STOP),)
SAMPLE_FLAGS += --stop "$(STOP)"
endif

TRAIN_FLAGS := --config $(CONFIG) --out $(CHECKPOINT) \
	--tokenizer $(TOKENIZER) --vocab-size $(VOCAB_SIZE) \
	--steps $(STEPS) --batch-size $(BATCH_SIZE) --seq-len $(SEQ_LEN) \
	--dim $(DIM) --layers $(LAYERS) --heads $(HEADS) --lr $(LR) \
	--warmup-steps $(WARMUP) --min-lr-ratio $(MIN_LR_RATIO) \
	--scheduler $(SCHEDULER) --grad-accum $(GRAD_ACCUM) \
	--log-every $(LOG_EVERY) --use-rope --use-gqa

ifneq ($(LOG_FILE),)
TRAIN_FLAGS += --log-file $(LOG_FILE)
endif

ifneq ($(RESUME),)
TRAIN_FLAGS += --resume $(RESUME)
endif

CHAT_FLAGS := --tokens $(CHAT_TOKENS) --temperature $(CHAT_TEMPERATURE) \
	--top-p $(CHAT_TOP_P) --repetition-penalty $(CHAT_REPETITION_PENALTY) \
	--template $(CHAT_TEMPLATE)
ifneq ($(CHAT_PROMPT),)
CHAT_FLAGS += --prompt "$(CHAT_PROMPT)"
endif
ifneq ($(CHAT_SYSTEM),)
CHAT_FLAGS += --system "$(CHAT_SYSTEM)"
endif
ifneq ($(CHAT_TOP_K),0)
CHAT_FLAGS += --top-k $(CHAT_TOP_K)
endif
ifneq ($(CHAT_STOP),)
CHAT_FLAGS += --stop "$(CHAT_STOP)"
endif
ifneq ($(CHAT_SEED),)
CHAT_FLAGS += --seed $(CHAT_SEED)
endif
ifeq ($(CHAT_STREAM),1)
CHAT_FLAGS += --stream
else
CHAT_FLAGS += --no-stream
endif
ifeq ($(CHAT_ONCE),1)
CHAT_FLAGS += --once
endif
ifeq ($(SHOW_PROMPT),1)
CHAT_FLAGS += --show-prompt
endif

.PHONY: all test data train resume sample chat eval clean

all: test train sample

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m tiny_llm.dataset \
		--raw data/raw --mixture data/mixture.json --out data/processed

train: data
	@mkdir -p checkpoints
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m tiny_llm.train $(TRAIN_DATA) $(TRAIN_FLAGS)

resume: data
	@test -n "$(RESUME)" || { echo "usage: make resume RESUME=checkpoints/tiny_bpe.pt STEPS=1200"; exit 2; }
	@mkdir -p checkpoints
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m tiny_llm.train $(TRAIN_DATA) $(TRAIN_FLAGS)

sample: $(CHECKPOINT)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m tiny_llm.sample $(CHECKPOINT) \
		--prompt "$(PROMPT)" --tokens $(TOKENS) --temperature $(TEMPERATURE) $(SAMPLE_FLAGS)

chat: $(CHECKPOINT)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m tiny_llm.chat $(CHECKPOINT) $(CHAT_FLAGS)

eval: $(CHECKPOINT) data
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m tiny_llm.eval $(CHECKPOINT) $(DATA) \
		--batch-size $(BATCH_SIZE)

clean:
	rm -rf .pytest_cache __pycache__ tests/__pycache__ src/tiny_llm/__pycache__
