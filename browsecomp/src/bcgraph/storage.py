"""Atomic results, append-only telemetry, and a durable request-result journal."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any
import uuid
from .evidence import stable_hash


def atomic_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class JournalConflict(RuntimeError):
    pass


class Store:
    """Use one Store per process/run. Do not run two processes in the same output dir."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta = self.path / "_meta"
        self.meta.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.meta / "journal.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS operations
            (key TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, status TEXT NOT NULL,
             value TEXT, updated REAL NOT NULL)""")
        self.db.commit()

    def get(self, key: str, fingerprint: str | None = None) -> dict | None:
        row = self.db.execute("SELECT fingerprint,status,value,updated FROM operations WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        if fingerprint is not None and fingerprint != row[0]:
            raise JournalConflict(f"Operation {key} replayed with different input; use a new attempt/run")
        return {"fingerprint": row[0], "status": row[1],
                "value": json.loads(row[2]) if row[2] else None, "updated": row[3]}

    def put(self, key: str, fingerprint: str, status: str, value: Any = None):
        old = self.get(key, fingerprint)
        if old and old["status"] == "success" and status != "success":
            raise JournalConflict(f"Cannot overwrite a committed operation: {key}")
        with self.db:
            self.db.execute("""INSERT INTO operations VALUES (?,?,?,?,?)
                ON CONFLICT(key) DO UPDATE SET status=excluded.status,value=excluded.value,updated=excluded.updated""",
                (key, fingerprint, status, json.dumps(value, ensure_ascii=False, allow_nan=False), time.time()))

    def operations(self, scope: str) -> list[dict]:
        prefixes = (scope + ":", "tool:" + scope + ":")
        rows = self.db.execute(
            "SELECT key,status,value FROM operations WHERE substr(key,1,?)=? OR substr(key,1,?)=?",
            (len(prefixes[0]), prefixes[0], len(prefixes[1]), prefixes[1]),
        ).fetchall()
        return [{"key": key, "status": status, "value": json.loads(value) if value else None}
                for key, status, value in rows]

    def event(self, kind: str, **fields: Any):
        record = {"kind": kind, "timestamp": time.time(), **fields}
        with (self.meta / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def result_path(self, query_id: str) -> Path:
        # Filenames cannot traverse outside the run directory.
        import base64
        encoded = base64.urlsafe_b64encode(query_id.encode()).decode().rstrip("=")
        return self.path / f"run_{encoded}.json"

    def write_result(self, value: dict):
        atomic_json(self.result_path(str(value["query_id"])), value)

    def close(self):
        self.db.close()


def source_code_hash() -> str:
    root = Path(__file__).parent
    return stable_hash({p.name: p.read_text(encoding="utf-8") for p in sorted(root.glob("*.py"))})


def redacted_config(config: dict) -> dict:
    value = deepcopy(config)
    value.get("retrieval", {})["env"] = {k: "<redacted>" for k in value.get("retrieval", {}).get("env", {})}
    return value
