"""Offline extraction is user-invoked; the implementation task does not execute it."""
import argparse
import json
import math
from pathlib import Path
import torch
from .config import ConsiSpaceConfig
from .features import FrozenDualEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="JSONL with question, frames, optional timestamps")
    parser.add_argument("--config", default="configs/consispace.json")
    parser.add_argument("--output", required=True, help="New directory for caches and manifest.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()
    root, destination = Path(args.data).resolve().parent, Path(args.output).resolve()
    if destination.exists() and any(destination.iterdir()):
        parser.error("Output directory must be empty to avoid overwriting feature caches")
    destination.mkdir(parents=True, exist_ok=True)
    config = ConsiSpaceConfig.load(args.config)
    encoder = FrozenDualEncoder(config, args.device, torch.bfloat16 if args.precision == "bf16" else torch.float32)
    with (destination / "manifest.jsonl").open("w") as manifest:
        for index, line in enumerate(Path(args.data).read_text().splitlines()):
            if not line.strip():
                continue
            record = json.loads(line)
            paths = [str((root / path).resolve()) for path in record["frames"]]
            features = encoder(paths, record["question"], record.get("timestamps"))
            geometry_unit = "native"
            if record.get("meters_per_unit") is not None:
                scale = float(record["meters_per_unit"])
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError("meters_per_unit must be a positive calibrated scale")
                features.poses[:, :3, 3] *= scale
                features.depth *= scale
                geometry_unit = "meters"
            cache = destination / f"scene_{index:08d}.pt"
            features.save(cache, record["question"], config, geometry_unit)
            record["features"] = cache.name
            record["frames"] = paths
            record["timestamp_unit"] = "seconds" if "timestamps" in record else "frame_index"
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()


if __name__ == "__main__":
    main()
