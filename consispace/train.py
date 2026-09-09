"""Explicitly invoked alignment / SFT / UC-SSRL entry point; no work at import."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from .config import ConsiSpaceConfig
from .data import SpatialDataset, prepare_record, single_record_collator
from .model import ConsiSpace


def export_checkpoint(accelerator, model, directory):
    """Gather only LoRA and bridge weights, including under ZeRO-3."""
    unwrapped = accelerator.unwrap_model(model)
    selected = {name: value for name, value in unwrapped.named_parameters()
                if name.startswith(("memory.", "aligner.")) or "lora_" in name}
    if accelerator.state.deepspeed_plugin is not None:
        import deepspeed
        context = deepspeed.zero.GatheredParameters(list(selected.values()), modifier_rank=0)
    else:
        context = nullcontext()
    with context:
        if accelerator.is_main_process:
            state = {name: value.detach().cpu().clone() for name, value in selected.items()}
            unwrapped.save_checkpoint(directory, state)
    accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["align", "sft", "ssrl"], required=True)
    parser.add_argument("--data", required=True, help="JSONL manifest with feature-cache paths")
    parser.add_argument("--config", default="configs/consispace.json")
    parser.add_argument("--checkpoint", help="Completed bridge/SFT checkpoint to initialize from")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attention", choices=["sdpa", "flash_attention_2", "eager"], default="sdpa")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--deepspeed", help="Optional ZeRO-3 config path")
    parser.add_argument("--resume-state", help="Accelerate optimizer/RNG state from the SAME stage")
    parser.add_argument("--save-state", action="store_true", help="Also save full optimizer/RNG state for exact resume")
    args = parser.parse_args()
    if args.steps <= 0 or args.gradient_accumulation <= 0 or args.learning_rate <= 0:
        parser.error("Steps, accumulation, and learning rate must be positive")
    if args.stage != "align" and not args.checkpoint:
        parser.error("SFT requires trained bridge initialization; UC-SSRL requires an SFT checkpoint")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume_state:
        parser.error("Output directory is nonempty; choose a new output or explicitly resume state")
    from accelerate import Accelerator, DeepSpeedPlugin, DataLoaderConfiguration
    from accelerate.utils import set_seed
    config = ConsiSpaceConfig.load(Path(args.checkpoint) / "config.json" if args.checkpoint else args.config)
    set_seed(config.seed)
    plugin = None
    if args.deepspeed:
        ds_config = json.loads(Path(args.deepspeed).read_text())
        ds_config["gradient_accumulation_steps"] = args.gradient_accumulation
        ds_config["bf16"] = {"enabled": args.precision == "bf16"}
        # Load small bridge/LoRA checkpoints before partitioning parameters.
        plugin = DeepSpeedPlugin(hf_ds_config=ds_config, zero3_init_flag=False)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation,
                              mixed_precision="bf16" if args.precision == "bf16" else "no",
                              deepspeed_plugin=plugin,
                              dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=config.seed))
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    dataset = SpatialDataset(args.data, args.stage)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=single_record_collator,
                        num_workers=args.workers, pin_memory=False,
                        generator=torch.Generator().manual_seed(config.seed))
    model = ConsiSpace(config, args.attention, dtype)
    if args.checkpoint:
        model.load_checkpoint(args.checkpoint)
    model.set_stage(args.stage)
    if accelerator.state.deepspeed_plugin is not None:
        from deepspeed.utils import set_z3_leaf_modules
        from .memory import GeometryConsistentMemory
        # Gating/fusion execute different numbers of key projections per scene.
        # Gather the small memory module as one leaf to keep collectives aligned.
        set_z3_leaf_modules(model, [GeometryConsistentMemory])
        if args.stage == "ssrl" and len({len(record["candidates"]) for record in dataset.records}) != 1:
            raise ValueError("ZeRO-3 UC-SSRL requires the same candidate count in all records; "
                             "bucket datasets by candidate count or use ordinary DDP")
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    unwrapped = accelerator.unwrap_model(model)
    completed, epoch, batches_in_epoch = 0, 0, 0
    if args.resume_state:
        accelerator.load_state(args.resume_state)
        progress = json.loads((Path(args.resume_state) / "progress.json").read_text())
        if progress["stage"] != args.stage:
            raise ValueError("Optimizer state belongs to a different stage")
        completed, epoch = progress["completed_steps"], progress["epoch"]
        batches_in_epoch = progress["batches_in_epoch"]
    if completed >= args.steps:
        raise ValueError("Requested total steps already completed by the resume checkpoint")
    optimizer.zero_grad()
    while completed < args.steps:
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        epoch_loader = accelerator.skip_first_batches(loader, batches_in_epoch) if batches_in_epoch else loader
        for record in epoch_loader:
            batch = prepare_record(record, unwrapped, accelerator.device, dtype)
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    result = model(**batch)
                    loss = result["loss"]
                if not bool(torch.isfinite(loss.detach())):
                    raise FloatingPointError("Non-finite loss; checkpoint was not exported")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            batches_in_epoch += 1
            if accelerator.sync_gradients:
                completed += 1
                metrics = {"step": completed, "stage": args.stage,
                           "loss": float(accelerator.reduce(loss.detach(), reduction="mean"))}
                accelerator.print(json.dumps(metrics))
                if completed >= args.steps:
                    break
        if completed < args.steps:
            epoch += 1
            batches_in_epoch = 0
    export_checkpoint(accelerator, model, output)
    if args.save_state:
        state_dir = output / "training_state"
        accelerator.save_state(str(state_dir))
        if accelerator.is_main_process:
            (state_dir / "progress.json").write_text(json.dumps({"stage": args.stage, "completed_steps": completed,
                                                               "epoch": epoch, "batches_in_epoch": batches_in_epoch}) + "\n")
    accelerator.end_training()


if __name__ == "__main__":
    main()
