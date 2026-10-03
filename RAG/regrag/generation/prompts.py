"""Prompt files, loaded once and identified by version + content hash.

WHY THIS EXISTS AT ALL — it replaces one line of `open(path).read()`.
    Because an Answer has to say HOW it was made, and "v3_json" alone is not
    enough: a prompt file can be edited without its name changing. The sha is
    what makes `config_hash` honest, and the reason is concrete.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
from dataclasses import dataclass
from functools import lru_cache

from regrag import config

PROMPT_DIR = config.PROJECT_ROOT / "prompts"
CURRENT = "answer_v3_json"

@dataclass(frozen=True)
class Prompt:
    version: str
    text: str
    sha: str            # first 16 hex of sha256 over the raw bytes

    def __str__(self) -> str:
        return f"<prompt {self.version} sha={self.sha} {len(self.text):,} chars>"


@lru_cache(maxsize=8)
def load(version: str = CURRENT) -> Prompt:
    path = PROMPT_DIR / f"{version}.md"
    if not path.exists():
        have = ", ".join(sorted(p.stem for p in PROMPT_DIR.glob("*.md")))
        raise FileNotFoundError(f"no prompt {version!r} in {PROMPT_DIR}. have: {have}")
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    # The whole citation design rests on the model naming a label the code can
    # resolve. A prompt that never mentions one cannot produce a traceable
    # answer, and the failure would surface much later as "no citations"
    # rather than here as "wrong prompt".

    if not re.search(r"\bS\d+\b", text):
        raise ValueError(f"{path} never names an S-label (S1, S2, ...) — it "
                         f"cannot ask for a citation this pipeline can resolve")
    return Prompt(version, text, hashlib.sha256(raw).hexdigest()[:16])
