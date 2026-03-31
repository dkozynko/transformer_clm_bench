from transformer_clm_bench.data import build_vocabulary, encode_tokens


def test_build_vocabulary_and_encoding_cover_special_tokens():
    vocab = build_vocabulary([["hello", "world"], ["hello"]], min_freq=1)
    ids = encode_tokens(["<bos>", "hello", "<eos>"], vocab)
    assert len(vocab) >= 5
    assert len(ids) == 3
