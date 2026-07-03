import argparse
import json

import torch
import ruamel.yaml as yaml
from PIL import Image
from torchvision import transforms

from models.model_re_mplug import MPLUG
from models.tokenization_bert import BertTokenizer
from models.vqa_critic import MPLUGCritic
from vqa_framework.schema import (
    ERROR_LABELS,
    NO_ERROR_LABEL,
    build_reformulation_input,
    build_vqa_input,
    multihot_to_labels,
)


def load_generator(model_path, config_path, device, max_length=50):
    config = yaml.load(open(config_path, "r", encoding="utf-8"), Loader=yaml.Loader)
    config["min_length"] = 1
    config["max_length"] = max_length
    config["beam_size"] = config.get("beam_size", 5)
    config["add_ocr"] = False
    config["add_object"] = False
    config.setdefault("text_encoder", "bert-base-multilingual-cased")
    config.setdefault("text_decoder", "bert-base-multilingual-cased")
    tokenizer = BertTokenizer.from_pretrained(config["text_encoder"])
    model = MPLUG(config=config, tokenizer=tokenizer)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    model.to(device)
    return model, tokenizer, config


def load_critic(model_path, config_path, device):
    config = yaml.load(open(config_path, "r", encoding="utf-8"), Loader=yaml.Loader)
    tokenizer = BertTokenizer.from_pretrained(config["text_encoder"])
    model = MPLUGCritic(config, num_labels=len(ERROR_LABELS))
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    model.to(device)
    return model, tokenizer, config


def build_transform(image_res):
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    return transforms.Compose(
        [
            transforms.Resize((image_res, image_res), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ]
    )


@torch.no_grad()
def generate(model, tokenizer, config, image, text, device):
    text_input = tokenizer(
        [text],
        padding="longest",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)
    topk_ids, _ = model(image, text_input, answer=None, train=False, k=config["k_test"])
    return (
        tokenizer.decode(topk_ids[0][0])
        .replace("[SEP]", "")
        .replace("[CLS]", "")
        .replace("[PAD]", "")
        .strip()
    )


@torch.no_grad()
def critique(model, tokenizer, config, image, question, answer, device):
    text = build_vqa_input(question, answer)
    text_input = tokenizer(
        [text],
        padding="longest",
        truncation=True,
        max_length=config.get("max_input_length", 96),
        return_tensors="pt",
    ).to(device)
    logits = model(image, text_input)
    probs = torch.sigmoid(logits)[0].detach().cpu().tolist()
    labels = multihot_to_labels(probs, threshold=config.get("threshold", 0.5))
    return labels, dict(zip(ERROR_LABELS, probs))


def run(args):
    device = torch.device(args.device)
    base, base_tokenizer, base_config = load_generator(
        args.base_model_path,
        args.base_config,
        device,
        max_length=args.max_answer_length,
    )
    reform, reform_tokenizer, reform_config = load_generator(
        args.reformulation_model_path,
        args.reformulation_config,
        device,
        max_length=args.max_answer_length,
    )
    critic, critic_tokenizer, critic_config = load_critic(
        args.critic_model_path,
        args.critic_config,
        device,
    )
    transform = build_transform(base_config["image_res"])
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

    answer = generate(base, base_tokenizer, base_config, image, args.question, device)
    history = []
    for step in range(args.max_rounds + 1):
        labels, scores = critique(
            critic,
            critic_tokenizer,
            critic_config,
            image,
            args.question,
            answer,
            device,
        )
        history.append({"round": step, "answer": answer, "feedback": labels, "scores": scores})
        if labels == [NO_ERROR_LABEL] or step == args.max_rounds:
            break
        reform_input = build_reformulation_input(args.question, answer, labels)
        answer = generate(reform, reform_tokenizer, reform_config, image, reform_input, device)
    return {"final_answer": answer, "history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--critic_model_path", required=True)
    parser.add_argument("--reformulation_model_path", required=True)
    parser.add_argument("--base_config", default="configs/re_mplug_base.yaml")
    parser.add_argument("--critic_config", default="configs/vqa_critic_base.yaml")
    parser.add_argument("--reformulation_config", default="configs/re_mplug_base.yaml")
    parser.add_argument("--max_rounds", type=int, default=2)
    parser.add_argument("--max_answer_length", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
