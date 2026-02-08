from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.types.events import AssistantThought, AssistantAction, CodeFragment


_THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_CODE_RE = re.compile(r"```(?P<lang>[a-zA-Z0-9_\-]+)?\n(?P<code>.*?)\n```", re.DOTALL)


@dataclass
class ParsedAssistant:
    thought: Optional[str]
    action: Optional[str]
    code_language: Optional[str]
    code: Optional[str]
    raw: str


def parse_structured_output(text: str) -> ParsedAssistant:
    """
    Parse LLM output that is expected to look like:

      <thought>...</thought>
      <action>...</action>
      ```python
      ...
      ```

    Returns a ParsedAssistant (fields may be None if missing).
    """
    thought_m = _THOUGHT_RE.search(text)
    action_m = _ACTION_RE.search(text)
    code_m = _CODE_RE.search(text)

    thought = thought_m.group(1).strip() if thought_m else None
    action = action_m.group(1).strip() if action_m else None

    code_language = None
    code = None
    if code_m:
        lang = code_m.group("lang") or "python"
        code_language = lang.strip() if lang else "python"
        code = code_m.group("code").rstrip()

    return ParsedAssistant(
        thought=thought,
        action=action,
        code_language=code_language,
        code=code,
        raw=text,
    )


def parsed_to_events(p: ParsedAssistant) -> List[object]:
    """
    Convert parsed assistant output to event objects (no ExecResult here).
    Policy:
      - If thought/action/code exist, emit them in that order.
    """
    events: List[object] = []
    if p.thought:
        events.append(AssistantThought(content=p.thought))
    if p.action:
        events.append(AssistantAction(content=p.action))
    if p.code is not None:
        events.append(CodeFragment(language=p.code_language or "python", code=p.code))
    return events
