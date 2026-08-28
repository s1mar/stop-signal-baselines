"""Common trajectory schema for liveness-lab (P2 slice).

A Trajectory is a sequence of Steps. Each `ai` step carries a parsed Action.
The representation is deliberately source-agnostic: Nebius SWE-agent rows and
locally generated runs both normalize into this.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# SWE-agent command categories
EDIT_CMDS = {"edit", "create"}
NAV_CMDS = {"open", "goto", "scroll_up", "scroll_down", "find_file", "search_file",
            "search_dir", "ls", "cat", "find"}
CONTROL_CMDS = {"submit"}


@dataclass
class Action:
    """A parsed SWE-agent command from an `ai` step."""
    raw: str                      # the raw command block (inside the fence)
    cmd: str                      # first token, e.g. "edit", "open", "python"
    args: str = ""                # remainder of the command line
    edit_content: str = ""        # for edit/create: the new text block written
    target_file: Optional[str] = None  # best-effort file the edit applies to

    @property
    def is_edit(self) -> bool:
        return self.cmd in EDIT_CMDS

    @property
    def is_nav(self) -> bool:
        return self.cmd in NAV_CMDS

    def normalized(self) -> str:
        """A normalized action string for syntactic-repetition detection.
        Collapses volatile literals so paraphrased-but-identical actions still
        differ (that is the point: syntactic detectors SHOULD miss paraphrase)."""
        return f"{self.cmd} {self.args}".strip().lower()


@dataclass
class Step:
    index: int                    # position within the (ai-only) action sequence
    raw_index: int                # position within the full role-tagged trajectory
    role: str                     # system | user | ai
    text: str
    action: Optional[Action] = None
    observation: str = ""         # the following `user` observation text, if any


@dataclass
class Trajectory:
    instance_id: str
    model_name: str
    resolved: bool
    steps: list[Step] = field(default_factory=list)   # ai steps only, in order
    generated_patch: str = ""
    gold_patch: str = ""
    exit_status: str = ""
    source: str = "nebius"

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def uid(self) -> str:
        """Stable per-trajectory id. The corpus has several trajectories per issue,
        so instance_id alone is not unique; this distinguishes them the way the
        mechanical detectors do (each trajectory is its own data point)."""
        return f"{self.instance_id}|{self.model_name}|{self.n_steps}|{len(self.generated_patch)}"

    @property
    def edit_steps(self) -> list[Step]:
        return [s for s in self.steps if s.action and s.action.is_edit]
