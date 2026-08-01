from __future__ import annotations
import argparse
import logging
from pathlib import Path

from .config import load_config


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="garl-trading")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run an experiment")
    run.add_argument("--config", default="configs/default.toml")
    report = subparsers.add_parser("report", help="rebuild a report from artifacts")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--confidence", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "run":
        from .experiment import ExperimentRunner

        config = load_config(args.config)
        run_dir = ExperimentRunner(config, args.config).run()
        print(run_dir)
    else:
        from .reporting.visualise import build_report

        build_report(Path(args.run_dir), confidence=args.confidence)

if __name__ == "__main__":
    main()
