from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Iterable, Optional

from src.types.session import Session
from src.types.events import AssistantThought


def session_to_jsonl(
    session: Session,
    *,
    include_thoughts: bool = False,
) -> str:
    """
    Export a Session to JSONL (one event per line).
    Useful for training / analytics pipelines.

    include_thoughts=False will omit AssistantThought events.
    """
    lines = []
    for ev in session.events:
        body = ev.body

        if (not include_thoughts) and isinstance(body, AssistantThought):
            continue

        rec = {
            "session_id": session.session_id,
            "event_id": ev.event_id,
            "type": body.__class__.__name__,
            "payload": _to_payload(body),
        }
        lines.append(json.dumps(rec, ensure_ascii=False))
    return "\n".join(lines)


def _to_payload(body: object) -> dict:
    if not is_dataclass(body):
        raise TypeError(f"Event body must be a dataclass instance, got {type(body)}")
    return asdict(body)
