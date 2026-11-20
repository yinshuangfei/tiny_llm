import json

import torch

from tiny_llm.scheduler import apply_lr_factor, lr_factor
from tiny_llm.train import build_parser, load_training_config, load_training_checkpoint, resolve_training_args, train


def test_lr_schedules_and_optimizer_update():
    assert lr_factor(1, 4, warmup_steps=2) == 0.5
    assert lr_factor(2, 4, warmup_steps=2) == 1.0
    assert lr_factor(4, 4, warmup_steps=0, scheduler="cosine") == 0.1
    assert lr_factor(4, 4, warmup_steps=0, scheduler="linear") == 0.1
    assert lr_factor(10, 4, scheduler="constant") == 1.0

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    apply_lr_factor(optimizer, 0.25)
    assert optimizer.param_groups[0]["lr"] == 0.25
    apply_lr_factor(optimizer, 0.5)
    assert optimizer.param_groups[0]["lr"] == 0.5


def test_training_config_overrides(tmp_path):
    path = tmp_path / "train.json"
    path.write_text(json.dumps({"steps": 7, "scheduler": "linear"}), encoding="utf-8")
    assert load_training_config(path) == {"steps": 7, "scheduler": "linear"}

    args = build_parser().parse_args(["data.txt", "--config", str(path), "--steps", "9"])
    resolved = resolve_training_args(args)
    assert resolved.steps == 9
    assert resolved.scheduler == "linear"
    assert resolved.batch_size == 8


def test_resume_gradient_accum_and_log(tmp_path):
    text = "abcdefgh" * 20
    checkpoint = tmp_path / "resume.pt"
    log = tmp_path / "train.jsonl"
    _, _, losses = train(
        text,
        steps=4,
        batch_size=2,
        seq_len=8,
        dim=16,
        layers=1,
        heads=2,
        warmup_steps=2,
        grad_accum=2,
        log_every=2,
        log_file=log,
        out=checkpoint,
    )
    assert len(losses) == 4
    assert checkpoint.exists()
    assert [json.loads(line)["step"] for line in log.read_text().splitlines()] == [2, 4]

    model, tokenizer, optimizer, step = load_training_checkpoint(checkpoint)
    assert step == 4
    assert optimizer.param_groups

    resumed_model, _, resumed_losses = train(
        text,
        steps=6,
        batch_size=2,
        seq_len=8,
        resume_path=checkpoint,
        warmup_steps=2,
        grad_accum=2,
    )
    assert len(resumed_losses) == 2
    _, _, _, final_step = load_training_checkpoint(checkpoint)
    assert final_step == 4

    train(
        text,
        steps=6,
        batch_size=2,
        seq_len=8,
        resume_path=checkpoint,
        warmup_steps=2,
        grad_accum=2,
        out=checkpoint,
    )
    _, _, _, final_step = load_training_checkpoint(checkpoint)
    assert final_step == 6
