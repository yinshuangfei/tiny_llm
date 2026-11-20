from tiny_llm.sample import build_parser


def test_parser_supports_stream_flag():
    parser = build_parser()
    args = parser.parse_args(["checkpoint.pt"])
    assert args.stream is False

    args = parser.parse_args(["checkpoint.pt", "--stream"])
    assert args.stream is True
