from __future__ import annotations

import asyncio
import json

import pytest

from topologist import Topologist
from topologist.streaming import EventStreamAdapter, RedisStreamAdapter


class RecordingStreamAdapter(EventStreamAdapter):
    async def connect(self) -> None:
        return None

    async def consume(self) -> None:
        return None


def test_event_stream_parses_bytes_and_processes_event() -> None:
    topo = Topologist()
    adapter = RecordingStreamAdapter(topo)

    event = adapter.parse_message(
        b'{"source": "Sensor", "relation": "observes", "target": "Signal", "confidence": 0.75}'
    )

    assert event["source"] == "Sensor"
    assert adapter.process_event(event)
    assert topo.neighbors("Sensor")[0]["confidence"] == pytest.approx(0.75)


def test_event_stream_rejects_invalid_event() -> None:
    adapter = RecordingStreamAdapter(Topologist())

    with pytest.raises(ValueError, match="source, relation, and target"):
        adapter.process_event({"source": "A", "relation": "", "target": "B"})


class FakeRedis:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []
        self.closed = False

    async def xread(
        self,
        streams: dict[str, str],
        block: int,
        count: int,
    ) -> list[tuple[bytes, list[tuple[bytes, dict[bytes, bytes]]]]]:
        self.calls.append(streams)
        assert block == 1000
        assert count == 10
        if len(self.calls) == 1:
            payload = json.dumps(
                {"source": "A", "relation": "streams_to", "target": "B", "confidence": 0.6}
            ).encode("utf-8")
            return [(b"topology:events", [(b"1-0", {b"data": payload})])]
        raise asyncio.CancelledError

    async def close(self) -> None:
        self.closed = True


def test_redis_consume_uses_message_id_as_next_offset() -> None:
    async def run() -> tuple[Topologist, FakeRedis]:
        topo = Topologist()
        redis = FakeRedis()
        adapter = RedisStreamAdapter(topo)
        adapter.redis = redis

        with pytest.raises(asyncio.CancelledError):
            await adapter.consume()
        await adapter.close()
        return topo, redis

    topo, redis = asyncio.run(run())

    assert topo.neighbors("A")[0]["relation"] == "streams_to"
    assert redis.calls == [{"topology:events": "0-0"}, {"topology:events": "1-0"}]
    assert redis.closed
