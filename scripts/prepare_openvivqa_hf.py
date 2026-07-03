import argparse
import json
from pathlib import Path

from datasets import load_dataset
from PIL import Image


QUESTION_CANDIDATES = ["question", "Question", "ques", "query", "prompt"]
ANSWER_CANDIDATES = ["answer", "Answer", "answers", "label", "gt_answer"]
IMAGE_CANDIDATES = ["image", "Image", "img", "image_path", "path", "filename"]
ID_CANDIDATES = ["question_id", "id", "qid", "qa_id"]


def choose_column(example, explicit, candidates, role):
    if explicit:
        if explicit not in example:
            raise KeyError(f"Column '{explicit}' for {role} is not in dataset columns")
        return explicit
    for name in candidates:
        if name in example:
            return name
    raise KeyError(
        f"Cannot infer {role} column. Available columns: {sorted(example.keys())}. "
        f"Pass --{role}_column explicitly."
    )


def normalize_answer(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if not value:
            return ""
        first = value[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return normalize_answer(first.get("answer") or first.get("text") or first)
    if isinstance(value, dict):
        if "answer" in value:
            return normalize_answer(value["answer"])
        if "text" in value:
            return normalize_answer(value["text"])
        if value:
            return str(max(value, key=value.get)) if all(isinstance(v, (int, float)) for v in value.values()) else str(value)
    return str(value)


def save_image(value, output_dir, index):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{index:08d}.jpg"

    if isinstance(value, Image.Image):
        value.convert("RGB").save(image_path)
        return str(image_path)

    if isinstance(value, str):
        src = Path(value)
        if src.exists():
            return str(src)
        raise FileNotFoundError(f"Image path from dataset does not exist locally: {value}")

    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and Path(value["path"]).exists():
            return str(Path(value["path"]))
        if value.get("bytes"):
            from io import BytesIO

            Image.open(BytesIO(value["bytes"])).convert("RGB").save(image_path)
            return str(image_path)

    raise TypeError(f"Unsupported image value type: {type(value)}")


def convert_split(args):
    dataset = load_dataset(args.dataset_name, args.config_name, split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    if len(dataset) == 0:
        raise ValueError("Selected split is empty")

    first = dataset[0]
    image_column = choose_column(first, args.image_column, IMAGE_CANDIDATES, "image")
    question_column = choose_column(first, args.question_column, QUESTION_CANDIDATES, "question")
    answer_column = choose_column(first, args.answer_column, ANSWER_CANDIDATES, "answer")
    id_column = None
    for candidate in ([args.id_column] if args.id_column else ID_CANDIDATES):
        if candidate and candidate in first:
            id_column = candidate
            break

    rows = []
    image_dir = Path(args.image_dir)
    for idx, sample in enumerate(dataset):
        rows.append(
            {
                "question_id": sample[id_column] if id_column else idx,
                "image": save_image(sample[image_column], image_dir, idx),
                "question": str(sample[question_column]),
                "answer": normalize_answer(sample[answer_column]),
            }
        )

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)

    print(f"Saved {len(rows)} samples to {output_file}")
    print(f"Columns: image={image_column}, question={question_column}, answer={answer_column}, id={id_column}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True, help="Hugging Face dataset id, for example owner/OpenViVQA")
    parser.add_argument("--config_name", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_file", default="data/openvivqa_train.json")
    parser.add_argument("--image_dir", default="data/openvivqa_images/train")
    parser.add_argument("--image_column", default="")
    parser.add_argument("--question_column", default="")
    parser.add_argument("--answer_column", default="")
    parser.add_argument("--id_column", default="")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()
    convert_split(args)
