# %%
from utils import *
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments
import argparse, os, json
from datasets import load_metric, load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
# %%

label_dicts = {
    "Gorilla_gorilla" : 0,
    "Homo_sapiens": 1,
} 
    
def define_argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_fn', default='armheb/DNA_bert_3')
    p.add_argument('--save_path', default='./model/')
    p.add_argument('--train_fn', required=True, default='../../data/train_prop.tsv')
    p.add_argument('--gradient_accumulation_steps', type=int, default=2)
    p.add_argument('--batch_size_per_device', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=5)
    p.add_argument('--warmup_ratio', type=float, default=.2)
    p.add_argument('--max_length', type=int, default=80)
    p.add_argument('--random_state', default=512, type=int)

    config = p.parse_args()

    return config
    
def train_model(config, train_dataset, valid_dataset, save_path):

    # You can change model here.
    tokenizer = AutoTokenizer.from_pretrained(config.model_fn)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_fn, num_labels=2)

    print(
            '|train| =', len(train_dataset),
            '|valid| =', len(valid_dataset),
    )

    total_batch_size = config.batch_size_per_device * torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) / total_batch_size * config.n_epochs)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)

    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(save_path, 'checkpoints'),
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        warmup_steps=n_warmup_steps,
        weight_decay=0.01,
        fp16=True,
        logging_steps=500,
        logging_strategy='steps',
        evaluation_strategy='steps',
        save_strategy='steps',
        metric_for_best_model='accuracy',
        greater_is_better=True,
        load_best_model_at_end=True,
        label_smoothing_factor=0.0,
        # save_steps=n_total_iterations // config.n_epochs,
        save_steps=500,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=dataCollator(tokenizer,
                                  config.max_length
                                  ),
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    trainer.train()
    trainer.model.save_pretrained(save_path)

    with open(os.path.join(save_path, 'args.json'), 'w') as f:
        json.dump(config.__dict__, f, indent=2)
        
def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def make_dataset(config, data):
    train, val = train_test_split(data, test_size=5000, stratify=data['label'])

    train_dataset = load_Dataset(
        train['k_mers'].to_list(),
        train['label'].to_list()
    )
    
    valid_dataset = load_Dataset(
        val['k_mers'].to_list(),
        val['label'].to_list()
    )
    return train_dataset, valid_dataset
        
if __name__ == '__main__':
    config = define_argparser()
    data = pd.read_csv(config.train_fn, delimiter='\t')
    
    train_dataset, valid_dataset = make_dataset(config, data)
    train_model(config, train_dataset, valid_dataset, config.save_path)