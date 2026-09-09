"""Score saved answers for VSI/OSI/MMSI-Video; never loads or runs a model."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean

from .evaluation.io import first_present, index_records, read_records
from .evaluation.metrics import (OSI_CHOICE, OSI_NUMERIC, VSI_CATEGORIES, VSI_CHOICE, VSI_NUMERIC,
                                 scalar_text, score_answer)


BENCHMARKS = {"vsi": "vsi", "vsibench": "vsi", "vsi-bench": "vsi",
              "osi": "osi", "osibench": "osi", "osi-bench": "osi",
              "mmsi": "mmsi", "mmsi-video": "mmsi", "mmsi-video-bench": "mmsi"}
SETTINGS = {"uniform-50": "uniform-50", "u50": "uniform-50", "sufficient-coverage": "sufficient-coverage",
            "sc": "sufficient-coverage"}
DIRECTION_TYPES = {"object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard"}


def setting_name(value):
    name = SETTINGS.get(str(value).strip().lower())
    if name is None:
        raise ValueError(f"Unknown MMSI-Video sampling setting: {value!r}")
    return name


def annotation_schema(row, benchmark, target_field="auto"):
    target = first_present(row, ("ground_truth", "answer", "target") if target_field == "auto" else (target_field,))
    if target is None:
        raise ValueError("Annotation is missing ground truth")
    category = first_present(row, ("category", "task", "question_type"))
    question_type = first_present(row, ("question_type", "category", "task"))
    if benchmark == "vsi":
        # The native VSI question_type is also the fine-grained task name.
        category = question_type
        if category in VSI_NUMERIC:
            kind = "numeric"
        elif category in VSI_CHOICE:
            kind = "choice"
        else:
            raise ValueError(f"Unknown VSI task: {category!r}")
    elif benchmark == "osi":
        if category in OSI_NUMERIC:
            kind = "numeric"
        elif category in OSI_CHOICE:
            kind = "choice"
        else:
            raise ValueError(f"Unknown OSI task: {category!r}")
        if question_type in {"numerical", "mcq"}:
            declared_kind = "numeric" if question_type == "numerical" else "choice"
            if declared_kind != kind:
                raise ValueError(f"OSI question_type contradicts category: {question_type!r}, {category!r}")
    else:
        category = category or "unclassified"
        text = scalar_text(target)
        kind = "choice" if text is not None and len(text) == 1 and text.isascii() and text.isalpha() else "exact"
    if not isinstance(category, str) or not category.strip():
        raise ValueError("Task category must be a nonempty string")
    return target, kind, category


def summary(rows):
    return {"count": len(rows), "score_percent": 100 * mean(row["score"] for row in rows) if rows else None,
            "missing_predictions": sum(row["missing_prediction"] for row in rows),
            "parse_errors": sum(row["parse_error"] and not row["missing_prediction"] for row in rows)}


def group_summary(rows, field):
    groups = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None and value != "":
            groups[str(value)].append(row)
    return {key: summary(group) for key, group in sorted(groups.items())}


def vsi_task_summary(by_category):
    tasks = dict(by_category)
    present = DIRECTION_TYPES.intersection(tasks)
    if present:
        if "object_rel_direction" in tasks:
            raise ValueError("Do not mix merged and difficulty-specific VSI direction annotations")
        selected = [tasks.pop(name) for name in sorted(present)]
        tasks["object_rel_direction"] = {
            "count": sum(group["count"] for group in selected),
            "score_percent": mean(group["score_percent"] for group in selected),
            "missing_predictions": sum(group["missing_predictions"] for group in selected),
            "parse_errors": sum(group["parse_errors"] for group in selected),
            "difficulty_groups_present": sorted(present),
            "difficulty_groups_missing": sorted(DIRECTION_TYPES - present),
        }
    missing = sorted(set(VSI_CATEGORIES) - tasks.keys())
    incomplete_direction = bool(present and present != DIRECTION_TYPES)
    return tasks, missing, incomplete_direction


def evaluate(annotations, predictions, benchmark, *, protocol="paper", answer_parser="strict",
             setting=None, annotation_id_field="auto", prediction_id_field="auto",
             target_field="auto", prediction_field="auto", allow_extra=False,
             require_all=False):
    if benchmark not in {"vsi", "osi", "mmsi"} or protocol not in {"paper", "benchmark"}:
        raise ValueError("Invalid benchmark or scoring protocol")
    if answer_parser not in {"strict", "final"}:
        raise ValueError("Invalid answer parser")
    if benchmark == "mmsi":
        if setting is None:
            raise ValueError("MMSI-Video requires an explicit sampling setting")
        setting = setting_name(setting)
    elif setting is not None:
        raise ValueError("--setting is only applicable to MMSI-Video")
    truth = index_records(annotations, annotation_id_field)
    guesses = index_records(predictions, prediction_id_field)
    if not truth:
        raise ValueError("Annotation set is empty")
    extra = sorted(guesses.keys() - truth.keys())
    missing = sorted(truth.keys() - guesses.keys())
    if extra and not allow_extra:
        raise ValueError(f"Predictions contain {len(extra)} unknown IDs; first IDs: {extra[:5]}")
    if missing and require_all:
        raise ValueError(f"Missing predictions for {len(missing)} IDs; first IDs: {missing[:5]}")
    rows = []
    for key, annotation in truth.items():
        guess = guesses.get(key)
        for record in (annotation, guess):
            if record is None:
                continue
            if record.get("benchmark") is not None:
                declared = BENCHMARKS.get(str(record["benchmark"]).lower())
                if declared != benchmark:
                    raise ValueError(f"ID {key}: record benchmark differs from selected benchmark")
            declared_setting = first_present(record, ("setting", "sampling_setting"))
            if declared_setting is not None and (benchmark != "mmsi" or setting_name(declared_setting) != setting):
                raise ValueError(f"ID {key}: mixed or contradictory sampling settings")
        target, kind, category = annotation_schema(annotation, benchmark, target_field)
        raw_prediction = None if guess is None else first_present(
            guess, ("prediction", "answer", "response") if prediction_field == "auto" else (prediction_field,))
        scored = score_answer(raw_prediction, target, kind, category, benchmark, protocol, answer_parser)
        row = {"id": key, "category": category, "kind": kind,
               "ground_truth": target, "prediction": raw_prediction, "missing_prediction": guess is None, **scored}
        for destination, fields in {
            "dataset": ("dataset", "source"), "scene": ("scene_name", "video_id", "video"),
            "difficulty": ("difficulty",), "question_type": ("question_type",),
        }.items():
            row[destination] = first_present(annotation, fields)
        rows.append(row)
    categories = group_summary(rows, "category")
    tasks, missing_tasks, incomplete_direction = (vsi_task_summary(categories) if benchmark == "vsi"
                                                  else (categories, [], False))
    micro = summary(rows)
    macro = mean(group["score_percent"] for group in tasks.values())
    aggregation = "task_macro_with_direction_difficulty_macro" if benchmark == "vsi" else "sample_micro"
    report = {"format_version": 1, "benchmark": benchmark, "protocol": protocol,
              "answer_parser": answer_parser, "setting": setting,
              "setting_verification": "declared_only_no_frame_inspection" if setting else None,
              "score_scale": "0-100 (per_sample.score uses 0-1)", "aggregation": aggregation,
              "overall": {**micro, "score_percent": macro if benchmark == "vsi" else micro["score_percent"]},
              "sample_micro_percent": micro["score_percent"], "task_macro_percent": macro,
              "by_task": tasks, "by_category": categories,
              "by_metric": group_summary(rows, "metric"), "by_dataset": group_summary(rows, "dataset"),
              "by_difficulty": group_summary(rows, "difficulty"), "by_scene": group_summary(rows, "scene"),
              "missing_prediction_ids": missing, "ignored_extra_ids": extra,
              "vsi_missing_tasks": missing_tasks, "vsi_incomplete_direction_groups": incomplete_direction,
              "annotation_scope": "caller_supplied_annotations_not_verified_as_full_benchmark"}
    return report, rows


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", nargs="+", required=True, help="One or more disjoint prediction shards")
    parser.add_argument("--output", required=True, help="JSON summary path (must not exist)")
    parser.add_argument("--details", help="Optional per-sample JSONL path (must not exist)")
    parser.add_argument("--protocol", choices=["paper", "benchmark"], default="paper")
    parser.add_argument("--answer-parser", choices=["strict", "final"], default="strict")
    parser.add_argument("--setting", choices=sorted(SETTINGS))
    parser.add_argument("--annotation-id-field", default="auto")
    parser.add_argument("--prediction-id-field", default="auto")
    parser.add_argument("--target-field", default="auto")
    parser.add_argument("--prediction-field", default="auto")
    parser.add_argument("--allow-extra-predictions", action="store_true")
    parser.add_argument("--require-all-predictions", action="store_true")
    args = parser.parse_args()
    destinations = [Path(path).resolve() for path in (args.output, args.details) if path]
    if len(set(destinations)) != len(destinations) or any(path.exists() for path in destinations):
        parser.error("Output paths must be distinct and must not exist")
    try:
        annotations = read_records(args.annotations)
        predictions = [row for path in args.predictions for row in read_records(path)]
        report, rows = evaluate(annotations, predictions, BENCHMARKS[args.benchmark], protocol=args.protocol,
                                answer_parser=args.answer_parser, setting=args.setting,
                                annotation_id_field=args.annotation_id_field, prediction_id_field=args.prediction_id_field,
                                target_field=args.target_field, prediction_field=args.prediction_field,
                                allow_extra=args.allow_extra_predictions, require_all=args.require_all_predictions)
        report["inputs"] = [{"path": str(Path(path).resolve()), "sha256": file_digest(path)}
                            for path in [args.annotations, *args.predictions]]
        # Serialize before opening output files; malformed/non-finite JSON cannot
        # leave a seemingly complete summary beside unfinished details.
        summary_text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        detail_text = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
        if args.details:
            with Path(args.details).open("x", encoding="utf-8") as stream:
                stream.write(detail_text)
        with Path(args.output).open("x", encoding="utf-8") as stream:
            stream.write(summary_text)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"benchmark": report["benchmark"], "setting": report["setting"],
                      "overall": report["overall"], "report": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
