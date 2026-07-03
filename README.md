# Image Captioning using Reformulations

The code in this repository is based on the original [mPLUG repository](https://github.com/alibaba/AliceMind/tree/main/mPLUG).

## Vietnamese VQA With Critic-Guided Reformulation

This fork adds a Vietnamese Visual Question Answering pipeline for OpenViVQA from Hugging Face, following the same idea as the original reformulation project: generate an initial answer, detect errors, then reformulate only when needed.

### Architecture

1. **Base VQA Model**
   - Input: image + Vietnamese question
   - Output: predicted answer
   - Implementation: reuse `models/model_re_mplug.py` as an image-text-to-text generator.

2. **Critic Model**
   - Input: image + question + predicted answer
   - Output: multi-label feedback:
     - Object Error
     - Attribute Error
     - Counting Error
     - Relation Error
     - Spatial Error
     - Causal Error
     - OCR/Text Error
     - Hallucination
     - Missing Detail
     - No Error
   - Implementation: `models/vqa_critic.py`.

3. **Reformulation Model**
   - Input: image + question + predicted answer + structured feedback
   - Output: corrected answer
   - Implementation: reuse `models/model_re_mplug.py`; the text input is built by `vqa_framework/schema.py`.

### Prepare OpenViVQA From Hugging Face

Convert the Hugging Face OpenViVQA split to the JSON format used by this repo:

```bash
python scripts/prepare_openvivqa_hf.py \
  --dataset_name <huggingface_openvivqa_dataset_id> \
  --split train \
  --output_file data/openvivqa_train.json \
  --image_dir data/openvivqa_images/train
```

If the Hugging Face columns have different names, pass them explicitly:

```bash
python scripts/prepare_openvivqa_hf.py \
  --dataset_name <huggingface_openvivqa_dataset_id> \
  --split train \
  --image_column image \
  --question_column question \
  --answer_column answer \
  --output_file data/openvivqa_train.json \
  --image_dir data/openvivqa_images/train
```

Repeat for validation/test splits, for example:

```bash
python scripts/prepare_openvivqa_hf.py \
  --dataset_name <huggingface_openvivqa_dataset_id> \
  --split validation \
  --output_file data/openvivqa_val.json \
  --image_dir data/openvivqa_images/val
```

### Project Data Format

OpenViVQA samples are normalized to this shape:

```json
{
  "question_id": 1,
  "image": "data/openvivqa_images/train/00000001.jpg",
  "question": "Trong ảnh có bao nhiêu người?",
  "answer": "hai"
}
```

The offline judging step writes:

- `critic_train.jsonl`: one JSON object per line with `image`, `question`, `predicted_answer`, `feedback`, and `corrected_answer`.
- `reformulation_train.jsonl`: JSONL copy for inspection.
- `reformulation_train.json`: JSON list compatible with the existing `train.py` reformulation loader.

### Offline Data Generation

Train the Base VQA model first with OpenViVQA using the existing training path. Set `train_file`, `val_file`, and `test_file` in `configs/vqa_base_mplug.yaml` to the converted OpenViVQA JSON files.

```bash
python train.py \
  --config configs/vqa_base_mplug.yaml \
  --checkpoint <pretrained_mplug_checkpoint> \
  --output_dir output_vqa_base \
  --text_encoder bert-base-multilingual-cased \
  --text_decoder bert-base-multilingual-cased
```

Then generate critic and reformulation data with a vision LLM judge:

```bash
python scripts/generate_vqa_feedback_data.py \
  --input_file data/openvivqa_train.json \
  --base_model_path output_vqa_base/checkpoint_07.pth \
  --output_dir data/openvivqa_feedback \
  --provider openai \
  --judge_model gpt-4o
```

For Gemini:

```bash
python scripts/generate_vqa_feedback_data.py \
  --input_file data/openvivqa_train.json \
  --base_model_path output_vqa_base/checkpoint_07.pth \
  --output_dir data/openvivqa_feedback \
  --provider gemini \
  --judge_model gemini-1.5-pro
```

Pass `--api_key` explicitly or set `OPENAI_API_KEY` / `GEMINI_API_KEY`.

### Train Critic

Set `train_file` and `val_file` in `configs/vqa_critic_base.yaml` to JSONL files, then run:

```bash
python train_vqa_critic.py \
  --config configs/vqa_critic_base.yaml \
  --output_dir output_vqa_critic
```

### Train Reformulation Model

Set `train_file`, `val_file`, and `test_file` in `configs/vqa_reformulation_mplug.yaml` to JSON list files such as `data/openvivqa_feedback/reformulation_train.json`, then run:

```bash
python train.py \
  --config configs/vqa_reformulation_mplug.yaml \
  --checkpoint <pretrained_mplug_checkpoint> \
  --output_dir output_vqa_reform \
  --text_encoder bert-base-multilingual-cased \
  --text_decoder bert-base-multilingual-cased
```

### Final Inference

```bash
python vqa_inference.py \
  --image path/to/image.jpg \
  --question "Trong ảnh có bao nhiêu người?" \
  --base_model_path output_vqa_base/checkpoint_07.pth \
  --critic_model_path output_vqa_critic/critic_checkpoint_04.pth \
  --reformulation_model_path output_vqa_reform/checkpoint_07.pth \
  --max_rounds 2
```

Inference flow:

```text
image + question
-> Base VQA predicted answer
-> Critic feedback
-> return answer if No Error
-> otherwise Reformulation corrects answer
-> Critic checks again
-> repeat up to N rounds
-> final answer
```

## Introduction

<img src="pipeline.jpg" width="400" height="400">

We propose an inference-time feedback model for the task of image captioning with a novel type of feedback, namely reformulation.
In the context of this type of feedback, the human annotators receive an image and a textual description as input, and subsequently produce text that is as similar as possible to the input text but also incorporates an additional desired attribute (e.g., improved factuality or a desired style). The reformulation model is trained to predict human reformulations.

## Installation
```
pip install -r requirements.txt
```

## Training
If you want to train your own reformulation model:
1. Download the mPLUG model from the original [repo](https://github.com/alibaba/AliceMind/tree/main/mPLUG).
2. Prepare a reformulation dataset in json format, i.e., a list of dictionaries, each with the following fields:
   - question_id: Any unique id
   - image: Path to image
   - question: Original caption
   - Answer: Reformulated caption
3. Modify the 'train_file' field in the relevant config file (configs/re_mplug_base.yaml or configs/re_mplug_large.yaml).
4. Train, e.g.:
```
bash scripts/re_mplug_base.sh 
```
Or:
```
bash scripts/re_mplug_large.sh 
```

We provide 3 reformulation datasets in this repository:
1. An error-correction reformulation data containing 5208 collected using Amazon Mechanical Turk (data/error_correction_dataset.json).
2. A style-transfer reformulation data transfering to humorous captions, harvested from FlickrStyle and the original Flickr30K captions (data/humor_train.json and data/humor_test.json).
3. A style-transfer reformulation data transfering to romantic captions, harvested from FlickrStyle and the original Flickr30K captions (data/romantic_train.json and data/romantic_test.json).

## Reformulate
To reformulate existing captions, create a json file containing a list of dictionaries, each with the following fields:
- caption: The caption to be reformulated
- image_path: Path to relevant image

Train your own reformulation model or download one of our provided models:

- [Error correction reformulation model](https://drive.google.com/drive/folders/1POjbnc7f3fHtve3y8wqQQvd-hQ-DwHhA?usp=sharing) (mPLUG base)
- [Humor style-transfer reformulation model](https://drive.google.com/file/d/1Un85hb6mdCjMA6cilfXUcwtaf29uyf25/view?usp=sharing) (mPLUG large)
- [Romantic style-transfer reformulation model](https://drive.google.com/file/d/1TThQIYb0G8PFut-fYGRV2WUkmfzfgjGd/view?usp=sharing) (mPLUG large)

Then, use the reformulate.py script.:
```
python reformulate.py --model_path <path to reformulation model> --mplug_backbone <base/large> --input_file <path to input json file> --output_file <path to output file, json format>
```

## Error Correction Examples

The following captions were generated by the [BLIP model](https://github.com/salesforce/BLIP) (BLIP) and reformualted using our error correction model (BLIP+re). We show improvement on different elements of the SPICE metric.

<img src="spice_improvement.jpg" width="889" height="500">

## Results on FlickrStyle

The following tables presents results of previous studies on the FlickrStyle dataset (top 3 rows), and the reformulated captions of the [BLIP model](https://github.com/salesforce/BLIP) (BLIP + Reformulation).

**Humorous**
|   | B@1 | B@3 | METEOR | CIDEr |
|:---:|:---:|:---:|:---:|:---:|
| [CapDec](https://arxiv.org/abs/2211.00575) | 29.4 | 8.8 | 13.2 | 55.1 |
| [SAN](https://dl.acm.org/doi/abs/10.1145/3474085.3475662?casa_token=eEuY_CIFt50AAAAA:1XO0pjvR1c4gc6rXIE3wOMXD6syJs7EjFx1vm6BH8i3x8hlMKvzBtWF6QGtDGvmmYWBImit1eiAl) | 29.5 | 9.9 | 12.5 | 47.2|
| [TridentCap](https://ieeexplore.ieee.org/abstract/document/10250859?casa_token=ZuWSXjb4e0cAAAAA:HPlMngIQepuYwn4PsZD43yQ89p-pTmocRF0MiiZEA3HT7swQLZV6nJyrcEFftZFbdtqDcsDWoQ) | 30.6 | 11.2 | 12.8 | 56.6 |
| BLIP + Reformulation | **33.7** | **11.7** | **14.8** | **72.0** |

The captions generated by BLIP + Reformulation can be found in generated_captions/style. Here are some examples:

<img src="style_examples.jpg" width="250" height="497">

## German Image Captioning

The code in this section is based on the [ClipCap repo](https://github.com/rmokady/CLIP_prefix_caption).

We provide a model for German image captioning, trained on the Multi30k dataset, and a pipeline for cross-lingual reformulation: generate a caption in German, translate to English, reformulate, translate back to German.

First, download our [German captioning model](https://drive.google.com/file/d/1LBCapDMsyRimYdkzHyAwRveqzKEGG2d2/view?usp=sharing).

### Generate German Captions

Use the german_captioning/inference.py script for generating German captions, e.g.
```
python german_captioning/inference.py --model_path <path to german model> --input_file <path to input file> --output_file <path to output file>
```
Where the input file is a list of dictionaries containing a single entry, 'image_path'.

If you want to use the reformulation pipeline, add the --reformulation_model flag with a path to the reformulation model, e.g.
```
python german_captioning/inference.py --model_path <path to german model> --input_file <path to input file> --output_file <path to output file> --reformulation_model error_correction_base.pth
```
Or
```
python german_captioning/inference.py --model_path <path to german model> --input_file <path to input file> --output_file <path to output file> --reformulation_model error_correction_large.pth --mplug_backbone large
```
This pipeline uses the [MarianMT open source translation models](https://huggingface.co/docs/transformers/model_doc/marian). If you prefer to use another translation API, you will need to generate the captions using the german_captioning/inference.py script, translate to English, reformulate using the reformulate.py script, and translate back to German.

### Results on Multi30k test set

The following table presents results of previous studies on the Multi30k test set (top 3 rows), our base model's performance (german_base.pt) and our reformulation pipeline performance (german_base.pt + Reformulation). For the bottom rows, we report standard deviation across 3 random initializations.

|   | B@4 | METEOR | CIDEr |
|:---:|:---:|:---:|:---:|
| [Dual Attention](https://aclanthology.org/W17-4750/) | 16.0 | 17.8 | 30.8 |
| [Cycle Consistency](https://ieeexplore.ieee.org/abstract/document/8784910?casa_token=daUUtFIIjMMAAAAA:1c2yKF7Qm0QHkkL58DdCgQnvnvD-4K46MS7IjjcHUrkZ-fnfVnyRatg-ffK3UW_uf0zWzeSHcA) | 15.9 | 17.8 | 31.0 |
| [Multi-Objective Optimization](https://dl.acm.org/doi/full/10.1145/3492325?casa_token=pWEip_CPb-AAAAAA%3ATeFSvodn14nbZYtYzJEf0-xtOkI9ETCliEQ3bh2YUN8lqQO7m474DiYiM_4jp9fXUGZg6CyIR4zk) | 16.5 | 17.9 | 33.8 |
| german_base.pt | 12.8 $\pm$ 0.3 | 18.6 $\pm$ 0.2 | 39.2 $\pm$ 1.6 |
| german_base.pt + Reformulation | **16.8 $\pm$ 0.1** | **20.1 $\pm$ 0.1** | **51.4 $\pm$ 0.6** |

The captions generated by german_base.pt + Reformulation can be found in generated_captions/german (each file contains the results for a single random initialization).
