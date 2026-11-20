PYTHON ?= python
PYTHONPATH ?= src
DATA ?= data/tiny.txt
CHECKPOINT ?= checkpoints/tiny_bpe.pt
CONFIG ?= configs/tiny_bpe.json
TOKENIZER ?= bpe
VOCAB_SIZE ?= 300
STEPS ?= 1000
BATCH_SIZE ?= 8
SEQ_LEN ?= 64
DIM ?= 64
LAYERS ?= 2
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

.PHONY: all test train resume sample eval clean

all: test train sample

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

train:
	@mkdir -p checkpoints
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m tiny_llm.train $(DATA) $(TRAIN_FLAGS)

resume:
	@test -n "$(RESUME)" || { echo "usage: make resume RESUME=checkpoints/tiny_bpe.pt STEPS=1200"; exit 2; }
	@mkdir -p checkpoints
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m tiny_llm.train $(DATA) $(TRAIN_FLAGS)

sample: $(CHECKPOINT)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m tiny_llm.sample $(CHECKPOINT) \
		--prompt "$(PROMPT)" --tokens $(TOKENS) --temperature $(TEMPERATURE) $(SAMPLE_FLAGS)

eval: $(CHECKPOINT)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m tiny_llm.eval $(CHECKPOINT) $(DATA) \
		--batch-size $(BATCH_SIZE)

clean:
	rm -rf .pytest_cache __pycache__ tests/__pycache__ src/tiny_llm/__pycache__
