from transformer_clm_bench.data import build_vocabulary, decode_token_ids, encode_text, encode_tokens


def test_build_vocabulary_and_encoding_cover_special_tokens():
    vocab = build_vocabulary([["hello", "world"], ["hello"]], min_freq=1)
    ids = encode_tokens(["<bos>", "hello", "<eos>"], vocab)
    assert len(vocab) >= 5
    assert len(ids) == 3


def test_byte_tokenization_round_trips_readable_text():
    token_ids = encode_text("Hello byte world!", tokenizer_name="byte")
    decoded = decode_token_ids(token_ids, tokenizer_name="byte")
    assert decoded == "Hello byte world!"
