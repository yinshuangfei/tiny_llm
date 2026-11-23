from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import torch

from tiny_llm.sample import _stop_cutoff, generate
from tiny_llm.train import load_checkpoint


@dataclass
class ChatSettings:
    max_new_tokens: int = 80
    temperature: float = 0.8
    top_k: int | None = None
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    seed: int | None = None


class ChatSession:
    """A small conversation state wrapper around the sampling API."""

    def __init__(
        self,
        model,
        tokenizer,
        settings: ChatSettings | None = None,
        *,
        template: str = "plain",
        system_prompt: str | None = None,
    ):
        if template not in {"plain", "chat"}:
            raise ValueError("template must be 'plain' or 'chat'")
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings or ChatSettings()
        self.template = template
        self.turns: list[tuple[str, str]] = []
        if system_prompt:
            self.turns.append(("system", system_prompt))

    def reset(self, system_prompt: str | None = None) -> None:
        self.turns.clear()
        if system_prompt:
            self.turns.append(("system", system_prompt))

    def add_user(self, text: str) -> None:
        self.turns.append(("user", text))

    def add_assistant(self, text: str) -> None:
        self.turns.append(("assistant", text))

    def build_prompt(self) -> str:
        if self.template == "plain":
            parts = [text for _, text in self.turns if text]
            if not parts:
                return ""
            if self.turns[-1][0] != "assistant":
                return parts[-1]
            return "\n".join(parts)

        parts = []
        for role, text in self.turns:
            if role == "system":
                parts.append(text)
            elif role == "user":
                parts.append(f"User: {text}")
            elif role == "assistant":
                parts.append(f"Assistant: {text}")
        if self.turns and self.turns[-1][0] == "user":
            parts.append("Assistant:")
        return "\n".join(parts)

    def respond(self, text: str, *, stream: bool = False, show_prompt: bool = False):
        if not text.strip():
            raise ValueError("prompt cannot be empty")
        self.add_user(text)
        prompt = self.build_prompt()
        if show_prompt:
            print(f"[prompt]\n{prompt}\n[response]", flush=True)

        if stream:
            reply = stream_reply(self.model, self.tokenizer, prompt, self.settings)
        else:
            full = generate(
                self.model,
                self.tokenizer,
                prompt,
                self.settings.max_new_tokens,
                self.settings.temperature,
                self.settings.top_k,
                self.settings.top_p,
                self.settings.repetition_penalty,
                self.settings.stop_sequences,
                self.settings.seed,
            )
            reply = full[len(prompt):].lstrip()
            print(reply, flush=True)

        self.add_assistant(reply)
        return reply


def stream_reply(model, tokenizer, prompt: str, settings: ChatSettings) -> str:
    """Stream only the newly generated text, without reprinting the prompt."""
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt produced no tokens")
    if settings.seed is not None:
        torch.manual_seed(settings.seed)

    generated: list[int] = []
    stop_sequences = settings.stop_sequences
    holdback = max((len(stop) for stop in stop_sequences), default=1) - 1
    # ``User:`` is 5 characters but often 1 BPE token. Matching against the
    # token count treats the prompt's own ``User:`` as a completed stop.
    prompt_length = len(tokenizer.decode(ids))
    printed_length = prompt_length

    for token in model.generate_stream(
        torch.tensor([ids], dtype=torch.long),
        settings.max_new_tokens,
        settings.temperature,
        settings.top_k,
        settings.top_p,
        settings.repetition_penalty,
    ):
        generated.append(token)
        text = tokenizer.decode(ids + generated)
        cutoff = _stop_cutoff(text, prompt_length, stop_sequences)
        if cutoff is not None:
            final_text = text[:cutoff]
            if len(final_text) > printed_length:
                print(final_text[printed_length:], end="", flush=True)
            print(flush=True)
            return final_text[prompt_length:].lstrip()

        safe_end = max(0, len(text) - holdback)
        if safe_end > printed_length:
            print(text[printed_length:safe_end], end="", flush=True)
            printed_length = safe_end

    final_text = tokenizer.decode(ids + generated)
    if len(final_text) > printed_length:
        print(final_text[printed_length:], end="", flush=True)
    print(flush=True)
    return final_text[prompt_length:].lstrip()


def _set_setting(settings: ChatSettings, key: str, value: str) -> str:
    if key == "tokens":
        settings.max_new_tokens = int(value)
    elif key == "temperature":
        settings.temperature = float(value)
    elif key == "top_k":
        settings.top_k = int(value)
    elif key == "top_p":
        settings.top_p = float(value)
    elif key == "repetition_penalty":
        settings.repetition_penalty = float(value)
    elif key == "stop":
        settings.stop_sequences = [value] if value else []
    elif key == "seed":
        settings.seed = int(value) if value else None
    else:
        return f"unknown setting: {key}"
    return f"set {key} = {value}"


def run_command(session: ChatSession, line: str) -> str | None:
    """Run a /command. Return a result marker, or None for normal chat text."""
    if line in {"/exit", "/quit"}:
        return "exit"
    if line == "/reset":
        system = next((text for role, text in session.turns if role == "system"), None)
        session.reset(system)
        return "reset"
    if line == "/settings":
        return str(session.settings)
    if line == "/help":
        return "commands: /exit /quit /reset /settings /set <name> <value>"
    if line.startswith("/set "):
        parts = line.split(" ", 2)
        if len(parts) != 3:
            return "usage: /set <name> <value>"
        return _set_setting(session.settings, parts[1], parts[2])
    if line.startswith("/"):
        return f"unknown command: {line.split()[0]}"
    return None


def build_parser():
    parser = argparse.ArgumentParser(description="Interactive tiny_llm chat/REPL")
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default=None, help="first prompt before entering the REPL")
    parser.add_argument("--system", default=None)
    parser.add_argument("--template", choices=["plain", "chat"], default="plain")
    parser.add_argument("--tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stream", dest="stream", action="store_true", default=True)
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--once", action="store_true", help="generate one response and exit")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    model, tokenizer = load_checkpoint(args.checkpoint)
    stop_sequences = list(args.stop)
    # Packed training documents are separated by a blank line. Stop there so
    # a short answer does not continue into the next document.
    if args.template == "chat" and "\n\n" not in stop_sequences:
        stop_sequences.append("\n\n")
    settings = ChatSettings(
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop_sequences=stop_sequences,
        seed=args.seed,
    )
    session = ChatSession(
        model,
        tokenizer,
        settings,
        template=args.template,
        system_prompt=args.system,
    )

    if args.prompt:
        print("assistant> ", end="", flush=True)
        session.respond(args.prompt, stream=args.stream, show_prompt=args.show_prompt)
        if args.once:
            return

    print("tiny_llm chat. Type /help for commands, /exit to quit.", flush=True)
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        command = run_command(session, line)
        if command == "exit":
            break
        if command is not None:
            print(command, flush=True)
            continue
        print("assistant> ", end="", flush=True)
        session.respond(line, stream=args.stream, show_prompt=args.show_prompt)


if __name__ == "__main__":
    main()
