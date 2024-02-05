from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch 
import sacrebleu
import logging
import sys 
logger = logging.getLogger("MainLogger")
logging.basicConfig(
    level=logging.INFO,
    format = "%(asctime)s|[%(levelname)s]|[%(name)s]|%(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
    stream= sys.stdout
)
def main(args):
    #1. load the dataset. flores 200 by default. 
    ds_config = f"{args.source_lang}-{args.target_lang}"
    ds = load_dataset(args.dataset_name, ds_config, split="dev",trust_remote_code=True)
    #2. load the model, config, tokenizer
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model).to(DEVICE)
    
    #3. generate predictions
    flores_lang_code= {
        'kor_Hang': 'Korean',
        'eng_Latn': 'English'
    }
    def add_prefix(x):
        x[f'sentence_{args.source_lang}'] = f'translate {flores_lang_code[args.source_lang]} to {flores_lang_code[args.target_lang]}: ' + x[f'sentence_{args.source_lang}']
        return x
    
    ds = ds.map(add_prefix)
    predictions = []
    references = []
    for data in tqdm(ds, total=len(ds), desc="Running prediction", ascii=' =', leave=True, position=0):
        input_ids = tokenizer(data[f'sentence_{args.source_lang}'], return_tensors='pt').input_ids.to(DEVICE)
        outputs = model.generate(input_ids)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction)
        references.append([data[f'sentence_{args.target_lang}']])
    
    #4. compute the metrics 
    metric = evaluate.load('sacrebleu')
    with open(f'./{args.model}_{args.dataset_name}_score.txt', mode="w+", encoding='utf-8') as f:
        for pred, ref in zip(predictions, references):
            score = metric.compute(predictions=pred, references=ref)['score']
            score = round(score, 6)
            f.write(f'{pred} :: {ref[0]} :: {score}\n')

    scores = 0
    with open(f'./{args.model}_{args.dataset_name}_score.txt', mode="r+", encoding='utf-8') as g:
        lines = g.readlines()
        for line in lines:
            _, _, score = line.split(" :: ")
            score = round(float(score[:-2]), 2)
            scores += score 
        logger.info(f"BLEU: {scores/len(lines)}")
            
            
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_arguments("--model", help="One of ['google/t5-base', 'google/t5-small', 'google/t5-large', 'facebook/nllb-200-distilled-600M', 'facebook/nllb-200-distilled-1.3B', 'facebook/nllb-200-3.3B', 'facebook/mbart-large-50-many-to-many-mmt']")
    parser.add_arguments("--dataset_name", default="facebook/flores")
    parser.add_arguments("--source_lang")
    parser.add_arguments("--target_lang")
    args = parser.parse_args()
    main(args)
    