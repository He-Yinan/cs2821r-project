from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Any, Dict


SCRATCH_ROOT = Path("/n/home13/yinan/cs2821r-project/results")

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]], mode: str = "w") -> None:
    ensure_dir(path.parent)
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_experiment_dir(experiment_name: str, *extra: Sequence[str | Path]) -> Path:
    base = SCRATCH_ROOT / "experiments" / experiment_name
    for component in extra:
        base = base / Path(component)
    ensure_dir(base)
    return base


