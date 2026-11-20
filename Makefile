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

.PHONY: all test train sample clean

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
		--prompt "$(PROMPT)" --tokens $(TOKENS) --temperature $(TEMPERATURE)

clean:
	rm -rf .pytest_cache __pycache__ tests/__pycache__ src/tiny_llm/__pycache__
