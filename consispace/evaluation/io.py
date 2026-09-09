"""Read local annotation/prediction files without importing model dependencies."""
import csv
import json
from pathlib import Path


def read_records(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise ValueError("Parquet inputs require: pip install -e '.[eval-data]'") from exc
        return parquet.read_table(path).to_pylist()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in stream if line.strip()]
        elif path.suffix.lower() == ".json":
            records = json.load(stream)
        elif path.suffix.lower() in {".csv", ".tsv"}:
            records = list(csv.DictReader(stream, delimiter="\t" if path.suffix.lower() == ".tsv" else ","))
        else:
            raise ValueError("Supported inputs: .jsonl, .json (array), .csv, .tsv, .parquet")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError(f"{path}: expected a list of records")
    return records


def record_id(record, field="auto"):
    fields = ("id", "question_id", "uid", "index") if field == "auto" else (field,)
    for name in fields:
        value = record.get(name)
        if value is not None and value != "":
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise ValueError(f"{name} must be a string or integer, got {value!r}")
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"{name} cannot be a whitespace-only ID")
            return str(value)
    raise ValueError(f"Missing stable sample ID (fields: {', '.join(fields)}); row-order matching is disabled")


def index_records(records, id_field="auto"):
    result = {}
    for row in records:
        key = record_id(row, id_field)
        if key in result:
            raise ValueError(f"Duplicate sample ID: {key!r}")
        result[key] = row
    return result


def first_present(record, fields):
    for field in fields:
        if record.get(field) is not None and record[field] != "":
            return record[field]
    return None
