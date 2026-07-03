import json
from typing import Dict, Iterable, List

ERROR_LABELS = [
    "Object Error",
    "Attribute Error",
    "Counting Error",
    "Relation Error",
    "Spatial Error",
    "Causal Error",
    "OCR/Text Error",
    "Hallucination",
    "Missing Detail",
    "No Error",
]

NO_ERROR_LABEL = "No Error"


def normalize_feedback(labels: Iterable[str]) -> List[str]:
    label_set = {label.strip() for label in labels if label and label.strip()}
    valid = [label for label in ERROR_LABELS if label in label_set]
    if not valid:
        return [NO_ERROR_LABEL]
    if NO_ERROR_LABEL in valid and len(valid) > 1:
        return [label for label in valid if label != NO_ERROR_LABEL]
    return valid


def labels_to_multihot(labels: Iterable[str]) -> List[int]:
    normalized = set(normalize_feedback(labels))
    return [1 if label in normalized else 0 for label in ERROR_LABELS]


def multihot_to_labels(values: Iterable[float], threshold: float = 0.5) -> List[str]:
    labels = [label for label, value in zip(ERROR_LABELS, values) if value >= threshold]
    return normalize_feedback(labels)


def build_vqa_input(question: str, predicted_answer: str = "") -> str:
    if predicted_answer:
        return f"Cau hoi: {question} [SEP] Cau tra loi du doan: {predicted_answer}"
    return f"Cau hoi: {question}"


def build_reformulation_input(
    question: str,
    predicted_answer: str,
    feedback: Iterable[str],
) -> str:
    feedback_text = "; ".join(normalize_feedback(feedback))
    return (
        f"Cau hoi: {question} [SEP] "
        f"Cau tra loi can kiem tra: {predicted_answer} [SEP] "
        f"Loi phat hien: {feedback_text}"
    )


def parse_json_object(text: str) -> Dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response does not contain a JSON object")
    return json.loads(text[start : end + 1])
