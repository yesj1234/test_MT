from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer 
import torch 
import sacrebleu
import logging
import sys 
import evaluate
import pandas as pd 
import numpy as np
from tqdm import tqdm 
logger = logging.getLogger("MainLogger")
logging.basicConfig(
    level=logging.INFO,
    format = "%(asctime)s|[%(levelname)s]|[%(name)s]|%(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
    stream= sys.stdout
)


def main(args):
    #1. load the dataset. flores 200 by default. 
    ds = load_dataset(args.dataset_name, f'{args.source_lang}-{args.target_lang}',split='dev', trust_remote_code=True)
    df = ds.to_pandas()
    #2. load the model, config, tokenizer
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(DEVICE)
    
    #3. generate predictions
   
    predictions = []
    references = []
    for data in tqdm(ds, total=len(ds), desc="Running prediction", ascii=' =', leave=True, position=0):
        input_ids = tokenizer(data[f'sentence_{args.source_lang}'], max_length=512, padding=True, return_tensors='pt').input_ids.to(DEVICE)
        outputs = model.generate(input_ids,forced_bos_token_id=tokenizer.lang_code_to_id[args.target_lang])
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction)
        references.append([data[f'sentence_{args.target_lang}']])
    assert len(predictions) == len(ds), f'length of predictions({len(predictions)}) does not match the length of ds({len(ds)})'
    assert len(transcriptions) == len(ds), f'length of predictions({len(transcriptions)}) does not match the length of ds({len(ds)})'
    # save the predictions in pd.DataFrame.
    DF_FILENAME = f'{args.model.replace("/", "_")}_{args.dataset_name.replace("/","_")}.csv'
    df.insert(len(ds), f'{args.model.replace("/", "_")}_prediction', predictions, True)
    df.to_csv(DF_FILENAME)
    #4. compute the metrics 
    metric = evaluate.load('sacrebleu')
    score_file_name = f"./{args.dataset_name.replace('/', '_')}_{args.model.replace('/','_')}_scores.txt" 
    scores = []
    with open(score_file_name, mode="w", encoding='utf-8') as f:
        for pred, ref in zip(predictions, references):
            score = metric.compute(predictions=[pred], references=ref)['score']
            score = round(score, 6)
            scores.append(score)
            f.write(f'{pred} :: {ref[0]} :: {score}\n')
    df.insert(len(ds), 'score', scores, True)
    df.to_csv(DF_FILENAME)
    scores = 0
    with open(score_file_name, mode="r", encoding='utf-8') as g:
        lines = g.readlines()
        for line in lines:
            _, _, score = line.split(" :: ")
            score = round(float(score[:-2]), 2)
            scores.append(score) 
        logger.info(f"BLEU: {sum(scores)/len(lines)}")
            
            
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="One of ['facebook/nllb-200-distilled-600M', 'facebook/nllb-200-distilled-1.3B', 'facebook/nllb-200-3.3B', 'facebook/mbart-large-50-many-to-many-mmt']")
    parser.add_argument("--dataset_name", default="facebook/flores")
    parser.add_argument("--source_lang")
    parser.add_argument("--target_lang")
    args = parser.parse_args()
    main(args)
    