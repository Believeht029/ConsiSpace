"""Deterministic scoring; protocol choices are recorded in each report."""
from decimal import Decimal, InvalidOperation
import math
import re


VSI_NUMERIC = {"object_abs_distance", "object_counting", "object_size_estimation", "room_size_estimation"}
VSI_CHOICE = {"object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard",
              "object_rel_direction", "object_rel_distance", "route_planning", "obj_appearance_order"}
OSI_NUMERIC = {"absolute_distance", "relative_direction_angular", "trajectory_length", "absolute_speed",
               "absolute_displacement", "object_3d_localization", "depth_aware_counting"}
OSI_CHOICE = {"relative_distance", "relative_direction_categorical", "relative_direction_categorical_cardinal",
              "relative_direction_categorical_ordinal", "trajectory_description"}
VSI_CATEGORIES = ("object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation",
                  "object_rel_distance", "object_rel_direction", "route_planning", "obj_appearance_order")
CONFIDENCES = tuple(Decimal(i) / 100 for i in range(50, 100, 5))
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def scalar_text(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return None
    return str(value).strip()


def numeric(value):
    text = scalar_text(value)
    if text is None or re.fullmatch(NUMBER, text) is None:
        return None
    try:
        result = Decimal(text)
    except InvalidOperation:
        return None
    return result if result.is_finite() else None


def final_text(text):
    """Only explicit final-answer containers; never search arbitrary reasoning for numbers."""
    tags = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    boxes = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if tags:
        return tags[-1].strip()
    if boxes:
        return boxes[-1].strip()
    lines = re.findall(r"^(?:final answer|answer|最终答案|答案)\s*[:：]\s*(.+?)\s*$", text, flags=re.I | re.M)
    return (lines[-1] if lines else text).strip().strip("`").strip()


def parse_prediction(value, kind, benchmark, protocol, parser):
    text = scalar_text(value)
    if text is None or not text:
        return None
    if parser == "final":
        text = final_text(text)
    if protocol == "benchmark" and benchmark == "vsi":
        text = text.split(" ")[0].rstrip(".").strip()
    elif protocol == "benchmark" and benchmark == "osi":
        # Official rule-based extraction, including its last-number convention.
        if kind == "numeric":
            numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", text.replace(",", ""))
            text = numbers[-1] if numbers else ""
        else:
            for pattern, source, flags in (
                (r'''[\(\[A-Z]\s*答案是?[:：]?\s*'?"?([A-Z])'?"?''', text, re.I),
                (r"[\[\(\s,.]([A-Z])[\]\)\s,.]", f" {text} ", 0),
                (r"^\s*([A-Z])", text, 0),
            ):
                match = re.search(pattern, source, flags)
                if match:
                    return match.group(1)
            return None
    if kind == "numeric":
        return numeric(text)
    if kind == "choice":
        # No substring matching: an option embedded in reasoning is not an answer.
        if re.fullmatch(r"[A-Za-z]", text):
            return text.upper() if benchmark != "mmsi" else text
        return None
    return text  # MMSI exact text matching when answers are not option letters.


def mra(prediction, target, protocol, zero_threshold=None):
    """Paper Eq. (18): ten confidence thresholds, strict relative-error comparison."""
    if target == 0:
        if protocol == "benchmark" and zero_threshold is not None:
            if prediction < zero_threshold:
                return 1.0, None
            target = zero_threshold
        else:
            return float(prediction == 0), None
    if protocol == "paper":
        error = abs(prediction - target) / abs(target)
        return sum(error < 1 - confidence for confidence in CONFIDENCES) / len(CONFIDENCES), str(error)
    # Preserve the upstream float/linspace point-count expression, rather than
    # silently changing it to the paper's strict Decimal threshold implementation.
    pred, truth = float(prediction), float(target)
    if not math.isfinite(pred) or not math.isfinite(truth) or truth == 0:
        return 0.0, None
    error = abs(pred - truth) / abs(truth)
    count = int((0.95 - 0.5) / 0.05 + 2)
    thresholds = [0.5 + i * ((0.95 - 0.5) / (count - 1)) for i in range(count)]
    thresholds[-1] = 0.95
    return sum(error <= 1 - confidence for confidence in thresholds) / count, str(error)


def score_answer(prediction, target, kind, category, benchmark, protocol="paper", parser="strict"):
    parsed = parse_prediction(prediction, kind, benchmark, protocol, parser)
    if kind == "numeric":
        truth = numeric(target)
        if truth is None or truth < 0:
            raise ValueError(f"Invalid nonnegative numerical ground truth: {target!r}")
        metric = "mra"
    else:
        truth = scalar_text(target)
        if not truth:
            raise ValueError("Missing or malformed ground truth")
        if kind == "choice" and re.fullmatch(r"[A-Za-z]", truth) is None:
            raise ValueError(f"Choice ground truth must be an option letter: {target!r}")
        if benchmark != "mmsi" and kind == "choice":
            truth = truth.upper()
        metric = "exact_match" if benchmark == "mmsi" else "accuracy"
    error = None
    if parsed is None:
        score = 0.0
    elif kind == "numeric":
        threshold = {"absolute_speed": Decimal("0.30"), "absolute_displacement": Decimal("0.30"),
                     "trajectory_length": Decimal("2.0")}.get(category) if benchmark == "osi" else None
        score, error = mra(parsed, truth, protocol, threshold)
    else:
        score = float(parsed == truth)
    return {"score": score, "metric": metric, "parsed_prediction": None if parsed is None else str(parsed),
            "parse_error": parsed is None, "relative_error": error}
