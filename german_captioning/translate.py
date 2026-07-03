from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import math
from tqdm import tqdm

def translate(data, source_language, target_language, batch_size=64):
    model_name = f'Helsinki-NLP/opus-mt-{source_language}-{target_language}'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)

    res = []
    batch_start = 0
    batch_num = math.ceil(len(data)/batch_size)
    for i in tqdm(range(batch_num), desc='Translation'):
        batch_start = i*batch_size
        batch_end = min(batch_start + batch_size, len(data))
        batch = [x['caption'] for x in data[batch_start:batch_end]]
        image_paths = [x['image_path'] for x in data[batch_start:batch_end]]
        inputs = tokenizer(batch, return_tensors='pt', padding=True).to(device)
        outputs = model.generate(**inputs, num_beams=5, num_return_sequences=1)
        res += [{'caption': tokenizer.decode(outputs[i], skip_special_tokens=True), 'image_path': image_paths[i]} for i in range(len(batch))]

    return res
