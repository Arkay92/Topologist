from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any

from topologist.engine import Topologist


class EventStreamAdapter(ABC):
    """Abstract event stream adapter for ingesting topology events."""

    def __init__(self, topology: Topologist) -> None:
        self.topology = topology

    @abstractmethod
    async def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def consume(self) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        return None

    def process_event(self, event: dict[str, Any]) -> bool:
        return self.topology.ingest_event(event)

    def parse_message(self, message: str | bytes) -> dict[str, Any]:
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        return json.loads(message)


class KafkaStreamAdapter(EventStreamAdapter):
    def __init__(self, topology: Topologist, bootstrap_servers: str = "localhost:9092", topic: str = "topology") -> None:
        super().__init__(topology)
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.consumer = None

    async def connect(self) -> None:
        try:
            from aiokafka import AIOKafkaConsumer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "aiokafka is required for Kafka streaming. "
                "Install with `pip install topologist[streaming]`."
            ) from exc
        self.consumer = AIOKafkaConsumer(self.topic, bootstrap_servers=self.bootstrap_servers)
        await self.consumer.start()

    async def consume(self) -> None:
        if self.consumer is None:
            raise RuntimeError("Kafka consumer has not been initialized.")
        async for msg in self.consumer:
            event = self.parse_message(msg.value)
            self.process_event(event)

    async def close(self) -> None:
        if self.consumer is not None:
            await self.consumer.stop()


class RedisStreamAdapter(EventStreamAdapter):
    def __init__(self, topology: Topologist, stream_key: str = "topology:events") -> None:
        super().__init__(topology)
        self.stream_key = stream_key
        self.redis = None

    async def connect(self) -> None:
        try:
            import redis.asyncio as redis_async  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "redis is required for Redis Streams ingestion. "
                "Install with `pip install topologist[streaming]`."
            ) from exc
        self.redis = redis_async.from_url("redis://localhost:6379")
        await self.redis.ping()

    async def consume(self) -> None:
        if self.redis is None:
            raise RuntimeError("Redis client has not been initialized.")
        last_id = "0-0"
        while True:
            messages = await self.redis.xread({self.stream_key: last_id}, block=1000, count=10)
            for _, entries in messages:
                for message_id, entry_dict in entries:
                    # extract data field from the entry dict
                    data = entry_dict.get(b"data", b"{}")
                    event = self.parse_message(data)
                    self.process_event(event)
                    # update last_id for next read
                    if isinstance(message_id, bytes):
                        last_id = message_id.decode("utf-8")
                    else:
                        last_id = message_id
            await asyncio.sleep(0.1)

    async def close(self) -> None:
        if self.redis is not None:
            await self.redis.close()


class WebSocketStreamAdapter(EventStreamAdapter):
    def __init__(self, topology: Topologist, uri: str = "ws://localhost:8000/events") -> None:
        super().__init__(topology)
        self.uri = uri
        self.connection = None

    async def connect(self) -> None:
        try:
            import websockets  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "websockets is required for WebSocket streaming. "
                "Install with `pip install topologist[streaming]`."
            ) from exc
        self.connection = await websockets.connect(self.uri)

    async def consume(self) -> None:
        if self.connection is None:
            raise RuntimeError("WebSocket connection has not been initialized.")
        async for message in self.connection:
            event = self.parse_message(message)
            self.process_event(event)

    async def close(self) -> None:
        if self.connection is not None:
            await self.connection.close()
