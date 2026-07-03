import sys
sys.path.append('.')
from predict import Predictor
import json
import argparse
from tqdm import tqdm
from reformulate import reformulate
from translate import translate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--reformulation_model', default=None)
    parser.add_argument('--mplug_backbone', default='base')
    parser.add_argument('--reformulation_batch_size', type=int, default=16)
    args = parser.parse_args()

    predictor = Predictor()
    predictor.setup(args.model_path)

    with open(args.input_file, 'r') as fp:
        dataset = json.load(fp)

    print('Dataset size: ' + str(len(dataset)), flush=True)
    res = []
    for sample in tqdm(dataset, desc='Generating captions'):
        image_path = sample['image_path']
        generated_caption = predictor.predict(image=image_path, use_beam_search=True)
        res.append({'image_path': image_path, 'caption': generated_caption})

    if args.reformulation_model is not None:
        orig_de_file = f'orig_de_{args.output_file}'
        with open(orig_de_file,  'w') as fp:
            json.dump(res, fp)

        print('Translating de->en...', flush=True)
        res = translate(res, 'de', 'en')
        orig_en_file = f'orig_en_{args.output_file}'
        with open(orig_en_file,  'w') as fp:
            json.dump(res, fp)

        print('Reformulating...', flush=True)
        res = reformulate(orig_en_file, args.reformulation_model, args.mplug_backbone, args.reformulation_batch_size)
        re_en_file = f're_en_{args.output_file}'
        with open(re_en_file,  'w') as fp:
            json.dump(res, fp)

        print('Translating en->de...', flush=True)
        res = translate(res, 'en', 'de')
    
    with open(args.output_file,  'w') as fp:
        json.dump(res, fp)

    print('Finished!')
