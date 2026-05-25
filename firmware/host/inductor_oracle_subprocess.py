"""Run TorchInductor ResNet-18 forwards in an isolated process (avoids NVRTC driver context clashes)."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Tuple


def run_inductor_oracle(
    seeds: Tuple[int, ...],
    input_size: int,
    weights_path: str | None = None,
) -> Dict[str, Any]:
    import torch
    import torchvision.models as models

    torch.backends.cudnn.enabled = False
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "cases": [
                {
                    "seed": int(s),
                    "status": "skipped",
                    "reason": "CUDA not available in inductor subprocess",
                }
                for s in seeds
            ],
        }

    device = torch.device("cuda")
    model = models.resnet18(weights=None).to(device).eval()
    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    compiled = torch.compile(model, backend="inductor", fullgraph=True)
    cases: List[Dict[str, Any]] = []
    for seed in seeds:
        entry: Dict[str, Any] = {"seed": int(seed)}
        try:
            gen = torch.Generator(device=device).manual_seed(int(seed))
            x = torch.randn(
                1, 3, input_size, input_size, generator=gen, device=device, dtype=torch.float32
            )
            with torch.no_grad():
                out = compiled(x)
            entry["status"] = "pass"
            entry["output"] = out.detach().cpu().numpy().tolist()
        except Exception as e:
            entry["status"] = "skipped"
            entry["reason"] = str(e)
        cases.append(entry)
    return {"cuda_available": True, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--seeds", default="0,1,42")
    parser.add_argument("--weights", default=None, help="torch.save state_dict path shared with parent benchmark")
    args = parser.parse_args()
    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    payload = run_inductor_oracle(
        seeds=seeds,
        input_size=int(args.input_size),
        weights_path=args.weights,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    failed = [c for c in payload["cases"] if c.get("status") != "pass"]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
