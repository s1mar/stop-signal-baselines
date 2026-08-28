"""Parse SWE-agent actions out of `ai` step text.

The agent's command sits in the final fenced code block of the step. Edit
commands look like:

    edit 22:25
    <new content lines>
    end_of_edit

`create <path>` opens a new empty file; a following `edit` writes into it.
`open <path>` / `create <path>` change the "current file", which is how we
attribute later edits to a file.
"""
from __future__ import annotations
import re
from typing import Optional
from .schema import Action

_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_EDIT_HEAD_RE = re.compile(r"^\s*edit\s+(\d+):(\d+)\s*$")


def extract_last_fence(text: str) -> Optional[str]:
    """Return the contents of the LAST fenced block, or None."""
    if not text:
        return None
    matches = _FENCE_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip("\n")


def parse_action(text: str) -> Optional[Action]:
    """Parse the SWE-agent command from one `ai` step's text (fenced block)."""
    fence = extract_last_fence(text)
    if fence is None:
        return None
    return parse_command(fence)


def _strip_root(path: str) -> str:
    for pre in ("/testbed/", "/testbed"):
        if path.startswith(pre):
            return path[len(pre):].lstrip("/")
    return path.lstrip("/")


def _parse_str_replace_editor(action: str) -> Action:
    """Parse the 2025 SWE-agent `str_replace_editor` command family:
    view/create/str_replace/insert, with content in --file_text / --new_str."""
    m = re.match(r"str_replace_editor\s+(\S+)\s+(\S+)", action)
    if not m:
        return Action(raw=action, cmd="str_replace_editor", args=action)
    sub, path = m.group(1), m.group(2)
    tf = _strip_root(path)
    content, cmd = "", sub
    if sub == "create":
        mm = re.search(r"--file_text\s+'(.*)'\s*$", action, re.DOTALL)
        content, cmd = (mm.group(1) if mm else ""), "create"
    elif sub in ("str_replace", "insert"):
        mm = re.search(r"--new_str\s+'(.*)'\s*$", action, re.DOTALL)
        content, cmd = (mm.group(1) if mm else ""), "edit"
    elif sub == "view":
        cmd = "open"
    return Action(raw=action, cmd=cmd, args=path, edit_content=content,
                  target_file=tf if cmd in ("edit", "create", "open") else None)


def parse_command(fence: str) -> Optional[Action]:
    """Parse a raw SWE-agent command block (already unwrapped, e.g. the `.traj`
    `action` field or the contents of a fenced block). Handles both the classic
    `edit start:end ... end_of_edit` format and the 2025 `str_replace_editor` family."""
    if fence is None:
        return None
    if fence.lstrip().startswith("str_replace_editor"):
        return _parse_str_replace_editor(fence.strip())
    lines = fence.split("\n")
    if not lines or not lines[0].strip():
        return None
    head = lines[0].strip()
    cmd = head.split()[0] if head.split() else ""
    args = head[len(cmd):].strip()

    edit_content = ""
    target_file = None

    if cmd == "edit":
        # content is everything after the head line up to (not incl.) end_of_edit
        body = lines[1:]
        if body and body[-1].strip() == "end_of_edit":
            body = body[:-1]
        else:
            body = [ln for ln in body if ln.strip() != "end_of_edit"]
        edit_content = "\n".join(body).strip("\n")
    elif cmd == "create":
        target_file = args.strip() or None
        # a bare `create <file>` may still be followed by inline content
        body = lines[1:]
        body = [ln for ln in body if ln.strip() != "end_of_edit"]
        edit_content = "\n".join(body).strip("\n")
    elif cmd in ("open",):
        target_file = args.split()[0] if args.split() else None

    return Action(raw=fence, cmd=cmd, args=args,
                  edit_content=edit_content, target_file=target_file)


def attach_current_file(actions: list[Action]) -> None:
    """Second pass: give each edit action the current open file (state machine).

    `open <path>` and `create <path>` set the current file. A plain `edit`
    inherits it. Mutates actions in place, setting `.target_file`.
    """
    current: Optional[str] = None
    for a in actions:
        if a is None:
            continue
        if a.cmd in ("open", "create") and a.target_file:
            current = a.target_file
        elif a.cmd == "edit" and a.target_file is None:
            a.target_file = current
