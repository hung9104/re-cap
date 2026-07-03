import os
import json
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from dataset.utils import pre_question

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class re_dataset(Dataset):
    def __init__(self, ann_file, transform, eos='[SEP]', split="train", max_ques_words=30, add_ocr=False, add_object=False):
        self.split = split        
        self.ann = []
        self.ann = json.load(open(ann_file,'r'))

        self.transform = transform
        self.max_ques_words = max_ques_words
        self.eos = eos
        self.add_ocr = add_ocr
        self.add_object = add_object
        
        if split=='test':
            self.max_ques_words = 50 # do not limit question length during test
        if self.add_ocr:
            self.max_ques_words = 30
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        ann = self.ann[index]
        
        image_path = ann['image']
            
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        question = ann['question']
        if self.add_ocr and "ocr" in ann:
            ocrs = ann['ocr']
            ocr_tokens = []
            poses = []
            for ocr in ocrs:
                pos, token = ocr
                ocr_tokens.append(token)
                poses.append(pos)
            if len(ocr_tokens) > 0:
                ocr_string = pre_question(" ".join(ocr_tokens), self.max_ques_words)
                question = question + " [SEP] " + ocr_string
        if self.add_object and "object_label" in ann:
            objects = ann["object_label"]
            question = question + " [SEP] " + " ".join(objects.split("&&"))
        # question = pre_question(question,self.max_ques_words)   
        if self.split == 'test':
            question_id = ann['question_id']
            return image, question, question_id

        elif self.split=='train':                                   
            answers = [ann['answer']]
            weights = [0.5]

            answers = [answer+self.eos for answer in answers]
                
            return image, question, answers, weights
