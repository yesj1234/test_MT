from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer 
import torch 
import sacrebleu
import logging
import sys 
import evaluate
import pandas as pd 
import numpy as np
import traceback 
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
        try:
            input_ids = tokenizer(data[f'sentence_{args.source_lang}'], max_length=512, padding=True, return_tensors='pt').input_ids.to(DEVICE)
            outputs = model.generate(input_ids,forced_bos_token_id=tokenizer.lang_code_to_id[args.target_lang])
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(prediction)
        except Exception as e:
            predictions.append("No Prediction")
            traceback.print_tb(e.__traceback__)
        references.append([data[f'sentence_{args.target_lang}']])
    assert len(predictions) == len(ds), f'length of predictions({len(predictions)}) does not match the length of ds({len(ds)})'
    assert len(references) == len(ds), f'length of predictions({len(references)}) does not match the length of ds({len(ds)})'

    DF_FILENAME = f'{args.model.replace("/", "_")}_{args.dataset_name.replace("/","_")}.csv'
    df.insert(len(df.columns), column=f'{args.model.replace("/", "_")}_prediction', value=predictions, allow_duplicates=True)

    #4. compute the metrics 
    metric = evaluate.load('sacrebleu')
    score_file_name = f"./{args.dataset_name.replace('/', '_')}_{args.model.replace('/','_')}_scores.txt" 
    scores = []
    for pred, ref in zip(predictions, references):
        score = metric.compute(predictions=[pred], references=ref)['score']
        score = round(score, 6)
        scores.append(score)
    df.insert(len(df.columns), column='score', value=scores, allow_duplicate=True)
    df.to_csv(DF_FILENAME)
            
            
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="One of ['facebook/nllb-200-distilled-600M', 'facebook/nllb-200-distilled-1.3B', 'facebook/nllb-200-3.3B', 'facebook/mbart-large-50-many-to-many-mmt']")
    parser.add_argument("--dataset_name", default="facebook/flores")
    parser.add_argument("--source_lang", help="eng_Latn, kor_Hang, zho_Hans, jpn_Jpan")
    parser.add_argument("--target_lang")
    args = parser.parse_args()
    main(args)
    
