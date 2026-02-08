from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from src.types.session import Session
from src.types.events import AssistantThought, EventBody

def session_to_jsonl(
    session: Session,
    *,
    include_thoughts: bool = False,
) -> str:
    """
    Export a Session to JSONL string (one JSON object per line).
    Ideal for:
      - Fine-tuning datasets (filtering out thoughts)
      - Analytics pipelines (ingesting into DBs)
    
    Args:
        session: The session to export.
        include_thoughts: If False, AssistantThought events are skipped.
                          (Useful for training concise models)
    """
    lines = []
    
    for ev in session.events:
        body = ev.body

        # 1. Privacy/Training Filter
        # If we want a concise dataset, skip the internal monologue.
        if (not include_thoughts) and isinstance(body, AssistantThought):
            continue

        # 2. Construct the record
        # We wrap the payload with metadata (ID, Type) for easier parsing later.
        rec = {
            "session_id": session.session_id,
            "event_id": ev.event_id,
            "type": body.__class__.__name__,
            "payload": _to_payload(body),
        }
        
        # 3. Serialize
        # ensure_ascii=False is critical for readable Korean text.
        lines.append(json.dumps(rec, ensure_ascii=False))

    return "\n".join(lines)


def _to_payload(body: Any) -> dict:
    """
    Convert a dataclass event body to a dict.
    """
    if not is_dataclass(body):
        # Should technically never happen due to strict typing in events.py
        raise TypeError(f"Event body must be a dataclass instance, got {type(body)}")
    
    # asdict is recursive, so nested dataclasses (like ExecutionOutput) work automatically.
    return asdict(body)
