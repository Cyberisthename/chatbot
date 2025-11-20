from jarvis5090x.infinite_cache import InfiniteMemoryCache


def test_cache_store_and_lookup():
    cache = InfiniteMemoryCache(max_items=10)
    signature = "op_signature"
    payload = {"value": 42}
    result = {"output": "computed"}

    assert cache.lookup(signature, payload) is None

    cache.store(signature, payload, result)
    cached = cache.lookup(signature, payload)
    assert cached == result


def test_cache_eviction():
    cache = InfiniteMemoryCache(max_items=3)
    payloads = [{"id": i} for i in range(4)]

    for payload in payloads:
        cache.store("op", payload, {"result": payload})

    assert cache.stats()["cached_items"] <= 3
