"""Turn diffs and agent edit-streams into comparable 'added-content' strings.

The oracle compares what the agent has WRITTEN so far against what the gold
patch ADDS. We deliberately compare added-content (not full file state), which
needs no repo checkout and is laptop-reproducible. See notes/feasibility.md for
why this approximation is used and how its error is calibrated.
"""
from __future__ import annotations
import re
from .schema import Trajectory

_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]{1,}")


def norm_line(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def identifiers(text: str) -> set[str]:
    """Code identifiers/keywords, robust to whitespace and formatting."""
    return set(_IDENT.findall(text or ""))


def gold_target_files(patch: str) -> set[str]:
    return set(gold_added_by_file(patch).keys())


def gold_added_tokens(patch: str) -> set[str]:
    """Identifiers appearing in the gold patch's added lines (>=3 char lines)."""
    toks: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            nl = norm_line(line[1:])
            if len(nl) >= 3:
                toks |= identifiers(nl)
    return toks


def agent_goldfile_states(traj: "Trajectory", gold_files: set[str]):
    """Per edit step: (step_index, cumulative_tokens_written_to_gold_files,
    cumulative_content_str). Only edits whose target file is a gold file count;
    this drops throwaway scaffolding (reproduce scripts, debug files)."""
    seen_tokens: set[str] = set()
    blocks: list[str] = []
    out = []
    for s in traj.steps:
        a = s.action
        if not (a and a.is_edit):
            continue
        if a.target_file in gold_files and a.edit_content:
            if not blocks or blocks[-1] != a.edit_content:
                blocks.append(a.edit_content)
            seen_tokens |= identifiers(a.edit_content)
        out.append((s.index, frozenset(seen_tokens), "\n".join(blocks)))
    return out


def gold_added_content(patch: str) -> str:
    """Concatenate added lines ('+' but not '+++') from a unified diff, in order.
    Grouped/ordered by file as they appear. Excludes diff metadata."""
    out: list[str] = []
    for line in patch.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            out.append(line[1:])
    return "\n".join(out).strip("\n")


def gold_added_by_file(patch: str) -> dict[str, str]:
    """Added content grouped per target file."""
    files: dict[str, list[str]] = {}
    cur = "<unknown>"
    for line in patch.splitlines():
        m = re.match(r"\+\+\+ b/(.+)$", line)
        if m:
            cur = m.group(1).strip()
            files.setdefault(cur, [])
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            files.setdefault(cur, []).append(line[1:])
    return {f: "\n".join(v).strip("\n") for f, v in files.items() if v}


def agent_cumulative_states(traj: Trajectory) -> list[tuple[int, str]]:
    """Return [(step_index, cumulative_added_content)] at each EDIT step.

    Per file, we keep an ordered list of edit blocks, skipping a block identical
    to that file's immediately-previous block (dedupe verbatim rewrites). The
    cumulative state at an edit step is the concatenation across files. This
    captures multi-region edits while tolerating the repeated rewrites that are
    the very phenomenon under study.
    """
    per_file: dict[str, list[str]] = {}
    states: list[tuple[int, str]] = []
    for s in traj.steps:
        a = s.action
        if not (a and a.is_edit):
            continue
        content = (a.edit_content or "").strip("\n")
        if not content:
            # a bare `create` with no content still advances state minimally
            if a.cmd == "create" and a.target_file:
                per_file.setdefault(a.target_file, [])
            # still record a state so step indexing stays aligned
        else:
            f = a.target_file or "<unknown>"
            blocks = per_file.setdefault(f, [])
            if not blocks or blocks[-1] != content:
                blocks.append(content)
        cumulative = "\n".join(
            "\n".join(blocks) for blocks in per_file.values() if blocks
        ).strip("\n")
        states.append((s.index, cumulative))
    return states
