import torch
from torch.utils.data import Dataset

class dataCollator():

    def __init__(self, tokenizer, max_length, inference=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inference = inference
        
    def __call__(self, samples):
        k_mers = [s['k_mers'] for s in samples]
        label = [s['label'] for s in samples]
        
        input_encoding = self.tokenizer(
            list(k_mers),
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.max_length
        )
        
        return_value = {
            'input_ids': input_encoding['input_ids'],
            'attention_mask': input_encoding['attention_mask'],
        }
        
        if not self.inference:
            encode_label = torch.tensor(list(map(lambda x: int(x), label)))
            return_value['labels'] = encode_label
            
        return return_value


class load_Dataset(Dataset):
    
    def __init__(self, k_mers, labels):
        self.k_mers = k_mers
        self.labels = labels
        
    def __len__(self):
        return len(self.k_mers)
    
    def __getitem__(self, item):
        k_mers = str(self.k_mers[item])
        label = str(self.labels[item])

        return {
            'k_mers': k_mers,
            'label': label,
        }