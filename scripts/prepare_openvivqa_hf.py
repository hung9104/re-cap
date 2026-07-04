import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image


QUESTION_CANDIDATES = ["question", "Question", "ques", "query", "prompt"]
ANSWER_CANDIDATES = ["answer", "Answer", "answers", "label", "gt_answer"]
IMAGE_CANDIDATES = ["image", "Image", "img", "image_path", "path", "filename"]
ID_CANDIDATES = ["question_id", "id", "qid", "qa_id"]
OPENVIVQA_FILES = {
    "train": ("vlsp2023_train_data.json", "train-images.zip"),
    "validation": ("vlsp2023_dev_data.json", "dev-images.zip"),
    "val": ("vlsp2023_dev_data.json", "dev-images.zip"),
    "dev": ("vlsp2023_dev_data.json", "dev-images.zip"),
    "test": ("vlsp2023_test_data.json", "test-images.zip"),
}


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


def first_present(record, names):
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return None


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


def flatten_records(data, split):
    if isinstance(data, list):
        return data
    for key in ["annotations", "data", "questions", "qas", split]:
        value = data.get(key) if isinstance(data, dict) else None
        if isinstance(value, list):
            return value
    if isinstance(data, dict):
        values = list(data.values())
        if values and all(isinstance(value, dict) for value in values):
            return values
    raise ValueError("Cannot find annotation records in OpenViVQA JSON")


def build_image_id_map(data):
    image_map = {}
    images = data.get("images", []) if isinstance(data, dict) else []
    if not isinstance(images, list):
        return image_map
    for item in images:
        if not isinstance(item, dict):
            continue
        image_id = first_present(item, ["id", "image_id", "img_id"])
        filename = first_present(item, ["file_name", "filename", "image", "image_path"])
        if image_id is not None and filename:
            image_map[str(image_id)] = str(filename)
    return image_map


def index_images(image_root):
    image_paths = {}
    for path in Path(image_root).rglob("*"):
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            continue
        image_paths[path.name] = path
        image_paths[path.stem] = path
        image_paths[str(path)] = path
    return image_paths


def resolve_image_path(record, image_id_map, image_paths):
    image_ref = first_present(
        record,
        ["image", "image_path", "img", "filename", "file_name", "image_id", "img_id"],
    )
    if image_ref is None:
        raise KeyError(f"Cannot find image reference in record keys: {sorted(record.keys())}")
    image_ref = str(image_ref)
    image_ref = image_id_map.get(image_ref, image_ref)
    candidates = [
        image_ref,
        Path(image_ref).name,
        Path(image_ref).stem,
        f"{image_ref}.jpg",
        f"{image_ref}.png",
    ]
    for candidate in candidates:
        if candidate in image_paths:
            return str(image_paths[candidate])
    raise FileNotFoundError(f"Cannot resolve image '{image_ref}' from annotation")


def convert_openvivqa_official(args):
    split_key = args.split.lower()
    if split_key not in OPENVIVQA_FILES:
        raise ValueError(f"Unsupported OpenViVQA split: {args.split}")

    ann_file, image_zip = OPENVIVQA_FILES[split_key]
    ann_path = hf_hub_download(
        repo_id=args.dataset_name,
        repo_type="dataset",
        filename=ann_file,
    )
    zip_path = hf_hub_download(
        repo_id=args.dataset_name,
        repo_type="dataset",
        filename=image_zip,
    )

    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    marker = image_dir / ".extracted"
    if not marker.exists():
        with ZipFile(zip_path, "r") as zip_fp:
            zip_fp.extractall(image_dir)
        marker.write_text("ok", encoding="utf-8")

    with open(ann_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    records = flatten_records(data, split_key)
    image_id_map = build_image_id_map(data)
    image_paths = index_images(image_dir)

    rows = []
    for idx, record in enumerate(records):
        if args.max_samples and len(rows) >= args.max_samples:
            break
        if not isinstance(record, dict):
            continue
        question = first_present(record, ["question", "Question", "ques", "query", "prompt"])
        if question is None:
            continue
        answer = first_present(record, ["answer", "Answer", "answers", "label", "gt_answer"])
        qid = first_present(record, ["question_id", "id", "qid", "qa_id"])
        rows.append(
            {
                "question_id": qid if qid is not None else idx,
                "image": resolve_image_path(record, image_id_map, image_paths),
                "question": str(question),
                "answer": normalize_answer(answer) if answer is not None else "",
            }
        )

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)
    print(f"Saved {len(rows)} OpenViVQA samples to {output_file}")


def convert_split(args):
    if args.openvivqa_official or args.dataset_name == "uitnlp/OpenViVQA-dataset":
        convert_openvivqa_official(args)
        return

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
    parser.add_argument("--openvivqa_official", action="store_true")
    args = parser.parse_args()
    convert_split(args)
