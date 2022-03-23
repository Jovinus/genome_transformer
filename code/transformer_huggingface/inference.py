# %%
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
from utils import *
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
import argparse
from sklearn.metrics import accuracy_score

def define_argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_fn', default='klue/roberta-large')
    p.add_argument('--trained_model', default='./model/klue_roberta-large/')
    p.add_argument('--save_path', default='./data/submission.csv')
    p.add_argument('--test_fn', required=True)
    p.add_argument('--valid_fn', required=True)
    p.add_argument('--device', default=0)

    config = p.parse_args()

    return config

def main(config):

    model = AutoModelForSequenceClassification.from_pretrained(config.model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(config.model_fn)
    test = pd.read_csv(config.test_fn, sep='\t', header=None, names=['text', 'label'])
    val = pd.read_csv(config.valid_fn, sep='\t', header=None, names=['text', 'label'])

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=config.device)
    out = classifier(test['text'].to_list())
    val_out = classifier(val['text'].to_list())

    label2int = {
        "Gorilla_gorilla" : 0,
        "Homo_sapiens": 1,
    }
    label2str = {
        "LABEL_0" : "Gorilla_gorilla" ,
        "LABEL_1" : "Homo_sapiens"
    }

    val_out= list(map(lambda x: x['label'], val_out))
    val['pred_label'] = val_out
    val['pred_label'] = val['pred_label'].replace(label2str)
    val['pred_label'] = val['pred_label'].replace(label2int)
    
    print("Validation Accuracy: %.3f"%accuracy_score(val['label'], val['pred_label']))
    
    out= list(map(lambda x: x['label'], out))
    test['label'] = out
    test['label'] = test['label'].replace(label2str)
    test['index'] = test.index 
    test[['index', 'label']].to_csv(config.save_path, index=False)

if __name__ == '__main__':
    config = define_argparser()
    main(config)