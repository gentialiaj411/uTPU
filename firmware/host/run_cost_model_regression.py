import argparse
import json
from pathlib import Path

from cost_model_regression import (
    DEFAULT_CALIBRATION_JSON,
    DEFAULT_OUTPUT_JSON,
    build_regression_report,
    replay_regression_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the frozen CUDA cost-model regression artifact under bench/results/."
    )
    parser.add_argument("--calibration-json", type=Path, default=DEFAULT_CALIBRATION_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    report = build_regression_report(
        calibration_json=args.calibration_json,
        output_json=args.output_json,
    )
    replay = replay_regression_report(report)
    print(json.dumps({"baseline": report["baseline"], "replay": replay}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
