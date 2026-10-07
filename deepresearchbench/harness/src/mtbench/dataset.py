from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_instance(dataset_path: Path, instance_id: int) -> dict[str, Any]:
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["id"]) == instance_id:
                return row
    raise KeyError(f"DeepResearchBench instance {instance_id} not found")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_prediction(path: Path, instance: dict[str, Any], article: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"id": instance["id"], "prompt": instance["prompt"], "article": article}
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

