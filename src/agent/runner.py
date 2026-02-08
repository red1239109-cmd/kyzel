from __future__ import annotations

import traceback as tb
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any

from src.types.events import ExecutionOutput


ApproveFn = Callable[[str, str], bool]  # (language, code) -> approved?


def default_approve(language: str, code: str) -> bool:
    """
    Simple console approval: y/n.
    """
    print("\n--- Proposed code to execute ---")
    print(f"[language={language}]")
    print(code)
    print("--- End code ---")
    ans = input("Execute? [y/N]: ").strip().lower()
    return ans == "y"


class PythonExecutor:
    """
    Minimal Python executor.
    - Runs code in a shared globals dict (state persists across steps).
    - Captures stdout/stderr naïvely by redirecting prints? (We keep it minimal.)
      For a production version, redirect sys.stdout/stderr to StringIO.
    """

    def __init__(self):
        self.globals: Dict[str, Any] = {}

    def run(self, code: str) -> ExecutionOutput:
        """
        Execute python code and return ExecutionOutput.
        stdout/stderr are simplified here.
        """
        # Minimal capture: we won't intercept prints unless you extend it.
        # We'll treat "no stdout/stderr capture" as empty strings per your policy.
        try:
            exec(code, self.globals, self.globals)
            return ExecutionOutput(stdout="", stderr="", success=True, traceback=None)
        except Exception as e:
            return ExecutionOutput(
                stdout="",
                stderr=str(e),
                success=False,
                traceback=tb.format_exc(),
            )
