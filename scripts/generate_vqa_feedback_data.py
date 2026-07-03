import argparse
import base64
import json
import os
from pathlib import Path

import torch
import requests
import ruamel.yaml as yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from models.model_re_mplug import MPLUG
from models.tokenization_bert import BertTokenizer
from vqa_framework.schema import (
    ERROR_LABELS,
    build_reformulation_input,
    normalize_feedback,
    parse_json_object,
)


def load_json_or_jsonl(path):
    with open(path, "r", encoding="utf-8") as fp:
        text = fp.read().strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def encode_image(path):
    with open(path, "rb") as fp:
        return base64.b64encode(fp.read()).decode("utf-8")


def build_judge_prompt(question, predicted_answer, reference_answer=None):
    reference = reference_answer or ""
    labels = ", ".join(ERROR_LABELS)
    return (
        "Bạn là bộ đánh giá Vietnamese Visual Question Answering.\n"
        "Hãy so sánh ảnh, câu hỏi, câu trả lời dự đoán và đáp án tham chiếu nếu có.\n"
        f"Các nhãn lỗi hợp lệ: {labels}.\n"
        "Trả về duy nhất JSON object với schema:\n"
        '{"feedback":["..."],"corrected_answer":"..."}\n\n'
        f"Câu hỏi: {question}\n"
        f"Câu trả lời dự đoán: {predicted_answer}\n"
        f"Đáp án tham chiếu: {reference}\n"
    )


def call_openai_vision(api_key, model, image_path, prompt):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def call_gemini_vision(api_key, model, image_path, prompt):
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": encode_image(image_path),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


def load_generator(model_path, mplug_backbone, device):
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    config = yaml.load(
        open(f"configs/re_mplug_{mplug_backbone}.yaml", "r", encoding="utf-8"),
        Loader=yaml.Loader,
    )
    config["min_length"] = 1
    config["max_length"] = 50
    config["beam_size"] = 5
    config["add_ocr"] = False
    config["add_object"] = False
    config["text_encoder"] = "bert-base-multilingual-cased"
    config["text_decoder"] = "bert-base-multilingual-cased"
    model = MPLUG(config=config, tokenizer=tokenizer)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    model.to(device)
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    transform = transforms.Compose(
        [
            transforms.Resize(
                (config["image_res"], config["image_res"]),
                interpolation=Image.BICUBIC,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return model, tokenizer, transform, config


@torch.no_grad()
def generate_answer(model, tokenizer, transform, config, image_path, question, device):
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    question_input = tokenizer(
        [question],
        padding="longest",
        truncation=True,
        max_length=96,
        return_tensors="pt",
    ).to(device)
    topk_ids, _ = model(image, question_input, answer=None, train=False, k=config["k_test"])
    return (
        tokenizer.decode(topk_ids[0][0])
        .replace("[SEP]", "")
        .replace("[CLS]", "")
        .replace("[PAD]", "")
        .strip()
    )


def main(args):
    data = load_json_or_jsonl(args.input_file)
    device = torch.device(args.device)
    base_model, tokenizer, transform, config = load_generator(
        args.base_model_path,
        args.mplug_backbone,
        device,
    )
    provider = args.provider.lower()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Pass --api_key or set provider env var.")

    critic_rows = []
    reform_rows = []
    for idx, row in enumerate(tqdm(data, desc="Build feedback data")):
        image_path = row.get("image") or row.get("image_path")
        question = row["question"]
        reference = row.get("answer") or row.get("answers") or row.get("label")
        if isinstance(reference, dict):
            reference = max(reference, key=reference.get)
        elif isinstance(reference, list):
            reference = reference[0] if reference else None

        predicted = generate_answer(
            base_model,
            tokenizer,
            transform,
            config,
            image_path,
            question,
            device,
        )
        prompt = build_judge_prompt(question, predicted, reference)
        if provider == "openai":
            raw = call_openai_vision(api_key, args.judge_model, image_path, prompt)
        elif provider == "gemini":
            raw = call_gemini_vision(api_key, args.judge_model, image_path, prompt)
        else:
            raise ValueError("--provider must be openai or gemini")
        judged = parse_json_object(raw)
        feedback = normalize_feedback(judged.get("feedback", []))
        corrected = judged.get("corrected_answer", reference or predicted)
        question_id = row.get("question_id", idx)

        critic_rows.append(
            {
                "question_id": question_id,
                "image": image_path,
                "question": question,
                "predicted_answer": predicted,
                "feedback": feedback,
                "corrected_answer": corrected,
            }
        )
        reform_rows.append(
            {
                "question_id": question_id,
                "image": image_path,
                "question": build_reformulation_input(question, predicted, feedback),
                "answer": corrected,
            }
        )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    write_jsonl(os.path.join(args.output_dir, "critic_train.jsonl"), critic_rows)
    write_jsonl(os.path.join(args.output_dir, "reformulation_train.jsonl"), reform_rows)
    with open(os.path.join(args.output_dir, "reformulation_train.json"), "w", encoding="utf-8") as fp:
        json.dump(reform_rows, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--provider", choices=["openai", "gemini"], default="openai")
    parser.add_argument("--judge_model", default="gpt-4o")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--mplug_backbone", default="base")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args)
