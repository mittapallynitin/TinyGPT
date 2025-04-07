import os

import requests
import torch
from torch.utils.data import DataLoader, Dataset


class CharDataset(Dataset):
    def __init__(self, data, max_seq=8):
        self.data = data
        self.max_seq = max_seq
        self.data_len = len(data)

    def __len__(self):
        return (self.data_len - self.max_seq)

    def __getitem__(self, idx) -> tuple[str, str]:        
        chunk = self.data[idx: idx + self.max_seq + 1]
        # Prepare x and y
        x = chunk[:-1]
        y = chunk[1:]
        
        return x, y

class CharDataLoader(DataLoader):
    def __init__(self, dataset, tokenizer, batch_size=8, shuffle=True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
        self.tokenizer = tokenizer

    def collate_fn(self, batch):
        x, y = zip(*batch)
        
        padded_x = []
        padded_y = []

        for seq_x, seq_y in zip(x, y):
            encoded_x, _ = self.tokenizer.encode(seq_x)
            encoded_y, _ = self.tokenizer.encode(seq_y)
            padded_x.append(torch.tensor(encoded_x))
            padded_y.append(torch.tensor(encoded_y))
            
        x = torch.stack(padded_x)
        y = torch.stack(padded_y)

        return x, y

def download_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    file_path = "./data/tinyshakespeare.txt"
    
    if not os.path.exists(file_path):
        print("Downloading Tiny Shakespeare dataset...")
        response = requests.get(url)
        with open(file_path, "w") as f:
            f.write(response.text)
    else:
        print("Dataset already downloaded.")
    
    with open(file_path, "r") as f:
        text = f.read()
    
    return text
