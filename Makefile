PYTHON ?= python
PYTHONPATH ?= src
DATA ?= data/tiny.txt
CHECKPOINT ?= checkpoints/tiny_bpe.pt
TOKENIZER ?= bpe
VOCAB_SIZE ?= 300
STEPS ?= 1000
SEQ_LEN ?= 64
DIM ?= 64
LAYERS ?= 2
HEADS ?= 4
LR ?= 3e-4
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

.PHONY: all test train sample eval clean

all: test train sample

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

$(CHECKPOINT): $(DATA)
	@mkdir -p $(@D)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m tiny_llm.train $(DATA) \
		--out $(CHECKPOINT) \
		--tokenizer $(TOKENIZER) --vocab-size $(VOCAB_SIZE) \
		--steps $(STEPS) --dim $(DIM) --layers $(LAYERS) --heads $(HEADS) \
		--batch-size 8 --seq-len $(SEQ_LEN) --lr $(LR) \
		--use-rope --use-gqa

train: $(CHECKPOINT)

sample: $(CHECKPOINT)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m tiny_llm.sample $(CHECKPOINT) \
		--prompt "$(PROMPT)" --tokens $(TOKENS) --temperature $(TEMPERATURE) $(SAMPLE_FLAGS)

eval: $(CHECKPOINT)
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m tiny_llm.eval $(CHECKPOINT) $(DATA) \
		--batch-size $(BATCH_SIZE)

clean:
	rm -rf .pytest_cache __pycache__ tests/__pycache__ src/tiny_llm/__pycache__
