from __future__ import annotations

from dataclasses import dataclass
from typing import List
from uuid import uuid4

from .events import EventBody


@dataclass
class SessionEvent:
    event_id: str
    body: EventBody

    @staticmethod
    def create(body: EventBody) -> "SessionEvent":
        return SessionEvent(event_id=str(uuid4()), body=body)


@dataclass
class Session:
    session_id: str
    events: List[SessionEvent]

    @staticmethod
    def create() -> "Session":
        return Session(session_id=str(uuid4()), events=[])

    def add(self, body: EventBody) -> SessionEvent:
        ev = SessionEvent.create(body)
        self.events.append(ev)
        return ev
