# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

tokenizer = AutoTokenizer.from_pretrained("armheb/DNA_bert_3")
# model = AutoModelForMaskedLM.from_pretrained("armheb/DNA_bert_3")
# %%
print(tokenizer("AAA ATG TTT", return_tensors='pt', padding=True))

print(tokenizer("aaa atg ttt", return_tensors='pt'))
# %%
model = AutoModelForSequenceClassification.from_pretrained("armheb/DNA_bert_3", num_labels=2)
# %%
tmp = ["AAA ATG TTT", "AAA TTT ATG AAA TTT"]
# %%
tokenizer(tmp, padding=True)
# %%

training_args = TrainingArguments(output_dir="../result/check_point/", 
                                  num_train_epochs=5, 
                                  per_device_train_batch_size=32, 
                                  per_device_eval_batch_size=32,
                                  warmup_steps='',
                                  fp16=True, 
                                  logging_strategy='epoch', 
                                  logging_dir="result.")

trainer = Trainer()