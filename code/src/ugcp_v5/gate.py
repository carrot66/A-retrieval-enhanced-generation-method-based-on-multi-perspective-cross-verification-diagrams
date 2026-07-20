"""Check that every declared dataset for one model strictly passes its target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_config = config["models"][args.model]
    rows = []
    for dataset in model_config["datasets"]:
        summary_path = (
            args.root
            / "artifacts"
            / "projections"
            / args.model
            / f"{dataset}.summary.json"
        )
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        accuracy = float(summary["metrics"]["full"]["acc"])
        target = float(model_config["targets"][dataset])
        rows.append(
            {
                "dataset": dataset,
                "acc": accuracy,
                "target": target,
                "margin": accuracy - target,
                "pass": accuracy > target,
            }
        )
    status = "PASS" if all(row["pass"] for row in rows) else "FAIL"
    result = {"model": args.model, "status": status, "results": rows}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise SystemExit(f"{args.model} gate failed")


if __name__ == "__main__":
    run()
