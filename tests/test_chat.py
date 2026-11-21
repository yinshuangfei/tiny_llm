import torch

from tiny_llm.chat import ChatSession, ChatSettings, build_parser, run_command
from tiny_llm.tokenizer import CharTokenizer
from tiny_llm.train import load_checkpoint, save_checkpoint
from tiny_llm.model import TinyCausalLM


def make_session(template="plain", system_prompt=None):
    text = "abcabc" * 12
    tokenizer = CharTokenizer.from_text(text)
    model = TinyCausalLM(
        vocab_size=tokenizer.vocab_size,
        dim=16,
        layers=1,
        heads=2,
        max_seq_len=16,
        use_rope=True,
        use_gqa=True,
    )
    model.eval()
    return ChatSession(
        model,
        tokenizer,
        ChatSettings(max_new_tokens=3, temperature=0, seed=1),
        template=template,
        system_prompt=system_prompt,
    )


def test_session_keeps_plain_history():
    session = make_session()
    first = session.respond("abc", stream=False)

    assert isinstance(first, str)
    assert session.turns == [("user", "abc"), ("assistant", first)]
    assert session.build_prompt() == "abc\n" + first

    session.add_user("abc")
    assert len(session.turns) == 3
    assert session.build_prompt() == "abc"


def test_chat_template_and_reset():
    session = make_session(template="chat")
    session.add_user("hello")
    prompt = session.build_prompt()
    assert prompt == "User: hello\nAssistant:"

    session.reset("System only")
    assert session.build_prompt() == "System only"


def test_settings_command():
    session = make_session()
    assert run_command(session, "/set temperature 0.2") == "set temperature = 0.2"
    assert session.settings.temperature == 0.2
    assert run_command(session, "/set top_k 0") == "set top_k = 0"
    assert session.settings.top_k == 0
    assert run_command(session, "/reset") == "reset"


def test_cli_once_with_saved_checkpoint(tmp_path):
    text = "abcabc" * 12
    tokenizer = CharTokenizer.from_text(text)
    model = TinyCausalLM(
        vocab_size=tokenizer.vocab_size,
        dim=16,
        layers=1,
        heads=2,
        max_seq_len=16,
        use_rope=True,
        use_gqa=True,
    )
    model.eval()
    path = tmp_path / "chat.pt"
    save_checkpoint(model, tokenizer, path)

    loaded_model, loaded_tokenizer = load_checkpoint(path)
    session = ChatSession(
        loaded_model,
        loaded_tokenizer,
        ChatSettings(max_new_tokens=2, temperature=0, seed=3),
    )
    reply = session.respond("abc", stream=False)
    assert isinstance(reply, str)
