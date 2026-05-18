from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


NANOGPT_REPO = Path("/home/zyh-ub/PyRepos/nanoGPT")


def run_nanogpt_train(
    config_path: str,
    extra_args: list[str] | None = None,
    nanogpt_repo: str | None = None,
) -> int:
    repo = Path(nanogpt_repo) if nanogpt_repo else NANOGPT_REPO
    cmd = [sys.executable, "train.py", config_path]
    if extra_args:
        cmd.extend(extra_args)
    print("running:", " ".join(shlex.quote(part) for part in cmd))
    completed = subprocess.run(cmd, cwd=repo)
    return completed.returncode
