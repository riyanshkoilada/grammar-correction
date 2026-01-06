import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class GrammarDataset(Dataset):
    def __init__(self, tokenizer, split='train', max_length=64, num_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load dataset
        # Using a subset of C4_200M for training to be faster
        # 'liweili/c4_200m' contains 'input' (incorrect) and 'output' (correct)
        print(f"Loading dataset {split}...")
        if split == 'train':
            # Using full C4_200M dataset
            # streaming=False because it failed in tests, and 15GB is manageable on disk.
            split_name = "train"
            if num_samples:
                split_name = f"train[:{num_samples}]"
            
            self.dataset = load_dataset("liweili/c4_200m", split=split_name, streaming=False)
            self.data = self.dataset
            
            if len(self.data) > 0:
                print(f"Dataset columns: {self.data[0].keys()}")
                
        elif split == 'validation':
            # Using JFLEG for validation
            self.dataset = load_dataset("jfleg", split="validation[:100]") # Small subset
            self.data = [{'input': item['sentence'], 'output': item['corrections'][0]} for item in self.dataset]
        
        print(f"Loaded {len(self.data)} samples for {split}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = item['input']
        tgt_text = item['output']

        # Tokenize
        src_tokens = self.tokenizer(src_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        tgt_tokens = self.tokenizer(tgt_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        src = src_tokens.input_ids.squeeze(0)
        tgt = tgt_tokens.input_ids.squeeze(0)
        
        # Create masks
        # src_mask: (1, 1, src_len)
        src_mask = (src != self.tokenizer.pad_token_id).unsqueeze(0).unsqueeze(0)
        
        # tgt_mask: (1, tgt_len, tgt_len)
        tgt_pad_mask = (tgt != self.tokenizer.pad_token_id).unsqueeze(0).unsqueeze(0)
        tgt_sub_mask = subsequent_mask(tgt.size(-1)).type_as(tgt_pad_mask)
        tgt_mask = (tgt_pad_mask & tgt_sub_mask).squeeze(1) # (1, S, S)

        # For training, we need tgt_y (target shifted by 1)
        # But for simplicity in the loop, we'll handle shifting there or return raw tensors
        # Standard Transformer training:
        # Input to decoder: tgt[:, :-1]
        # Target for loss: tgt[:, 1:]
        
        return {
            'src': src,
            'tgt': tgt,
            'src_mask': src_mask,
            'tgt_mask': tgt_mask,
            'src_text': src_text,
            'tgt_text': tgt_text
        }

def collate_batch(batch):
    # Custom collate if needed, but default should work since we pad to max_length
    # Just stacking tensors
    src = torch.stack([item['src'] for item in batch])
    tgt = torch.stack([item['tgt'] for item in batch])
    src_mask = torch.stack([item['src_mask'] for item in batch])
    tgt_mask = torch.stack([item['tgt_mask'] for item in batch])
    
    return {
        'src': src,
        'tgt': tgt,
        'src_mask': src_mask, # (B, 1, 1, S) -> (B, 1, S) adjustment needed?
        # src_mask shape in __getitem__ is (1, 1, S). Stack -> (B, 1, 1, S).
        # Transformer expects (B, 1, 1, S) or (B, 1, S, S) broadcastable.
        # Let's keep it as is.
        'tgt_mask': tgt_mask,
        'src_text': [item['src_text'] for item in batch],
        'tgt_text': [item['tgt_text'] for item in batch]
    }
