from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .config import ExperimentConfig, load_config
from .runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mtbench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run one LangGraph benchmark instance")
    run.add_argument("--config", required=True)
    run.add_argument("--run-id")
    run.add_argument("--planner-max-tokens", type=int,
                     help="Planner output budget; overrides config.run.planner_max_tokens")
    run.add_argument("--review-max-tokens", type=int,
                     help="Review output budget; overrides config.run.review_max_tokens")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    data = load_config(args.config).model_dump()
    for field in ("planner_max_tokens", "review_max_tokens"):
        value = getattr(args, field, None)
        if value is not None:
            data["run"][field] = value
    # Validate CLI overrides too; run_experiment persists the effective config.
    return ExperimentConfig.model_validate(data)


async def async_main(args: argparse.Namespace) -> int:
    if args.command == "run":
        root = Path(__file__).resolve().parents[2]
        run_dir, _ = await run_experiment(root, config_from_args(args), run_id=args.run_id)
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        print(json.dumps({"run_dir": str(run_dir), **status}, ensure_ascii=False))
        return 0
    raise AssertionError(args.command)


def main() -> None:
    raise SystemExit(asyncio.run(async_main(build_parser().parse_args())))


if __name__ == "__main__":
    main()
