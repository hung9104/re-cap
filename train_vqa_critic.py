import argparse
import datetime
import os
import random
import time
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

import utils
from dataset.randaugment import RandomAugment
from dataset.vqa_feedback_dataset import VQACriticDataset
from models.tokenization_bert import BertTokenizer
from models.vqa_critic import MPLUGCritic
from optim import create_optimizer
from scheduler import create_scheduler
from vqa_framework.schema import ERROR_LABELS


def create_critic_loaders(config):
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config["image_res"],
                scale=(0.5, 1.0),
                interpolation=Image.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            RandomAugment(
                2,
                7,
                isPIL=True,
                augs=[
                    "Identity",
                    "AutoContrast",
                    "Equalize",
                    "Brightness",
                    "Sharpness",
                    "ShearX",
                    "ShearY",
                    "TranslateX",
                    "TranslateY",
                    "Rotate",
                ],
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (config["image_res"], config["image_res"]),
                interpolation=Image.BICUBIC,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_set = VQACriticDataset(config["train_file"], train_transform)
    val_set = VQACriticDataset(config["val_file"], test_transform)
    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size_train"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size_test"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, tokenizer, optimizer, scheduler, epoch, device, config):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=50, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = f"Critic Train Epoch: [{epoch}]"
    warmup_iterations = config["schedular"]["warmup_epochs"] * 100
    for i, (image, text, labels) in enumerate(metric_logger.log_every(loader, 50, header)):
        image = image.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        text_input = tokenizer(
            list(text),
            padding="longest",
            truncation=True,
            max_length=config.get("max_input_length", 96),
            return_tensors="pt",
        ).to(device)
        loss, _ = model(image, text_input, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch == 0 and i % 100 == 0 and i <= warmup_iterations:
            scheduler.step(i // 100)
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, config):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    threshold = config.get("threshold", 0.5)
    total = 0
    exact = 0
    tp = torch.zeros(len(ERROR_LABELS), device=device)
    fp = torch.zeros(len(ERROR_LABELS), device=device)
    fn = torch.zeros(len(ERROR_LABELS), device=device)
    for image, text, labels in metric_logger.log_every(loader, 50, "Critic Eval:"):
        image = image.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        text_input = tokenizer(
            list(text),
            padding="longest",
            truncation=True,
            max_length=config.get("max_input_length", 96),
            return_tensors="pt",
        ).to(device)
        logits = model(image, text_input)
        pred = (torch.sigmoid(logits) >= threshold).float()
        exact += (pred == labels).all(dim=1).sum().item()
        total += labels.size(0)
        tp += (pred * labels).sum(dim=0)
        fp += (pred * (1 - labels)).sum(dim=0)
        fn += ((1 - pred) * labels).sum(dim=0)
    precision = (tp.sum() / (tp.sum() + fp.sum()).clamp_min(1)).item()
    recall = (tp.sum() / (tp.sum() + fn.sum()).clamp_min(1)).item()
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    stats = {"exact": exact / max(total, 1), "micro_f1": f1}
    print("Critic stats:", stats)
    return stats


def main(args, config):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    tokenizer = BertTokenizer.from_pretrained(config["text_encoder"])
    model = MPLUGCritic(config, num_labels=len(ERROR_LABELS))
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        print(f"load checkpoint from {args.checkpoint}")
    device = torch.device(args.device)
    model.to(device)

    optimizer = create_optimizer(utils.AttrDict(config["optimizer"]), model)
    scheduler, _ = create_scheduler(utils.AttrDict(config["schedular"]), optimizer)
    train_loader, val_loader = create_critic_loaders(config)

    start_time = time.time()
    for epoch in range(config["schedular"]["epochs"]):
        if epoch > 0:
            scheduler.step(epoch + config["schedular"]["warmup_epochs"])
        train_one_epoch(model, train_loader, tokenizer, optimizer, scheduler, epoch, device, config)
        evaluate(model, val_loader, tokenizer, device, config)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "config": config,
                "epoch": epoch,
                "labels": ERROR_LABELS,
            },
            os.path.join(args.output_dir, f"critic_checkpoint_{epoch:02d}.pth"),
        )
    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {elapsed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vqa_critic_base.yaml")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output_dir", default="output_vqa_critic")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    config = yaml.load(open(args.config, "r", encoding="utf-8"), Loader=yaml.Loader)
    yaml.dump(config, open(os.path.join(args.output_dir, "config.yaml"), "w"))
    main(args, config)
