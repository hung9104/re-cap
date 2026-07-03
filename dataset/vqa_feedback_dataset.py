import json

import torch
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

from vqa_framework.schema import build_vqa_input, labels_to_multihot

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class VQACriticDataset(Dataset):
    def __init__(self, ann_file, transform):
        self.ann = []
        with open(ann_file, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    self.ann.append(json.loads(line))
        self.transform = transform

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        image = Image.open(ann["image"]).convert("RGB")
        image = self.transform(image)
        text = build_vqa_input(ann["question"], ann["predicted_answer"])
        labels = labels_to_multihot(ann.get("feedback", ["No Error"]))
        return image, text, torch.tensor(labels, dtype=torch.float32)
