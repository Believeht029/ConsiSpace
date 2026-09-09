"""Generate answers with an explicitly supplied completed checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from .config import ConsiSpaceConfig
from .data import SpatialDataset, prepare_record
from .model import ConsiSpace
from .evaluation.io import index_records, record_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--attention", choices=["sdpa", "flash_attention_2", "eager"], default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Output already exists")
    dataset = SpatialDataset(args.data, "predict")
    # Ensure exported answers can be joined to benchmark annotations unambiguously.
    index_records(dataset.records)
    config = ConsiSpaceConfig.load(Path(args.checkpoint) / "config.json")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model = ConsiSpace(config, args.attention, dtype)
    model.load_checkpoint(args.checkpoint)
    model.set_stage("predict")
    model.to(args.device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("x") as output:
        for record in dataset:
            batch = prepare_record(record, model, args.device, dtype)
            answer, retrieval = model.generate_answer(batch["scene"], batch["prompt_ids"], batch["answer_ids"],
                                                       args.max_new_tokens, args.temperature)
            result = {"id": record_id(record), "question": record["question"], "answer": answer,
                      "retrieval": retrieval}
            for field in ("benchmark", "setting", "sampling_setting"):
                if field in record:
                    result[field] = record[field]
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
