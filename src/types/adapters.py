from __future__ import annotations

from typing import List, Dict

from src.types.session import Session
from src.types.events import HumanMsg, AssistantThought, AssistantAction, CodeFragment, ExecResult


ChatMessage = Dict[str, str]
Conversation = List[ChatMessage]


def session_to_conversation(
    session: Session,
    *,
    include_thoughts: bool = False,
    include_exec_results: bool = False,
) -> Conversation:
    conv: Conversation = []
    buf: List[str] = []

    def flush_assistant():
        nonlocal buf
        if buf:
            conv.append({"role": "assistant", "content": "\n".join(buf).strip()})
            buf = []

    for ev in session.events:
        body = ev.body

        if isinstance(body, HumanMsg):
            flush_assistant()
            conv.append({"role": "user", "content": body.content})

        elif isinstance(body, AssistantThought):
            if include_thoughts:
                buf.append(f"<thought>{body.content}</thought>")

        elif isinstance(body, AssistantAction):
            buf.append(f"<action>{body.content}</action>")

        elif isinstance(body, CodeFragment):
            lang = body.language or "python"
            buf.append(f"```{lang}\n{body.code}\n```")

        elif isinstance(body, ExecResult):
            if include_exec_results:
                out = body.output
                block = [
                    f"<exec success={out.success}>",
                    "stdout:",
                    out.stdout or "",
                    "stderr:",
                    out.stderr or "",
                ]
                if out.traceback:
                    block += ["traceback:", out.traceback]
                block.append("</exec>")
                buf.append("\n".join(block))

        else:
            # ResumeFrom 등은 프롬프트에 직접 넣지 않는 편이 안전
            pass

    flush_assistant()
    return conv
