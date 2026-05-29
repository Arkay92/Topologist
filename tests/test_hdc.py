from topologist.hdc import HyperVectorSpace


def test_stable_item_memory() -> None:
    hdc = HyperVectorSpace(dim=1024, seed=7)
    first = hdc.get("Memory")
    second = hdc.get("Memory")
    assert (first == second).all()


def test_similarity_self_is_one() -> None:
    hdc = HyperVectorSpace(dim=1024, seed=7)
    vector = hdc.get("A")
    assert hdc.similarity(vector, vector) == 1.0


def test_relation_encoding_has_expected_dim() -> None:
    hdc = HyperVectorSpace(dim=1024, seed=7)
    relation = hdc.encode_relation("A", "causes", "B")
    assert relation.shape == (1024,)


def test_confidence_encoding_varies() -> None:
    hdc = HyperVectorSpace(dim=1024, seed=7)
    low = hdc.encode_confidence(0.1)
    high = hdc.encode_confidence(0.9)
    assert low.shape == (1024,)
    assert high.shape == (1024,)
    # different confidence values should not be identical
    assert hdc.similarity(low, high) < 0.9999
