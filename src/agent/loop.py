from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable

from src.model.llm import LLM
from src.types.session import Session
from src.types.events import (
    HumanMsg,
    CodeFragment,
    ExecResult,
)
from src.agent.parser import parse_structured_output, parsed_to_events
from src.agent.runner import PythonExecutor, default_approve, ApproveFn


@dataclass
class AgentConfig:
    max_new_tokens: int = 512
    temperature: float = 1.0
    include_thoughts_in_prompt: bool = False
    include_exec_results_in_prompt: bool = False


class AgentLoop:
    """
    Session-first agent loop:
      - user writes HumanMsg -> append to Session
      - LLM generates structured output -> parse -> append thought/action/code events
      - if code exists -> ask approval -> execute -> append ExecResult
      - repeat
    """

    def __init__(
        self,
        llm: LLM,
        *,
        approve_fn: ApproveFn = default_approve,
        executor: Optional[PythonExecutor] = None,
        config: Optional[AgentConfig] = None,
    ):
        self.llm = llm
        self.approve_fn = approve_fn
        self.executor = executor or PythonExecutor()
        self.config = config or AgentConfig()

    def step(self, session: Session) -> Session:
        """
        One agent step: generate -> parse -> append -> maybe execute.
        """
        raw = self.llm.generate(
            session,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            include_thoughts_in_prompt=self.config.include_thoughts_in_prompt,
            include_exec_results_in_prompt=self.config.include_exec_results_in_prompt,
        )

        parsed = parse_structured_output(raw)
        new_events = parsed_to_events(parsed)

        # Append parsed assistant events
        for e in new_events:
            session.add(e)

        # If code present, ask approval and execute
        code_ev = next((e for e in new_events if isinstance(e, CodeFragment)), None)
        if code_ev is not None and code_ev.language.lower() == "python":
            if self.approve_fn(code_ev.language, code_ev.code):
                out = self.executor.run(code_ev.code)
                session.add(ExecResult(output=out))

        return session

    def run_cli(self) -> Session:
        """
        Simple CLI driver.
        """
        session = Session.create()
        while True:
            user_text = input("\nUser> ").strip()
            if not user_text:
                continue
            if user_text in {"/quit", "/exit"}:
                break

            session.add(HumanMsg(content=user_text))
            self.step(session)

        return session
