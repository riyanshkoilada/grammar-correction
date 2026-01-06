import torch
from grammar_correction.model import make_model

def test_shapes():
    vocab_size = 1000
    d_model = 512
    model = make_model(vocab_size, vocab_size, N=2, d_model=d_model)
    
    batch_size = 8
    seq_len = 64
    
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2) # (B, 1, 1, S)
    tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2) # (B, 1, 1, S)
    
    # Decoder input and target
    decoder_input = tgt[:, :-1]
    target = tgt[:, 1:]
    tgt_mask = tgt_mask[:, :, :-1, :-1] # (B, 1, S-1, S-1)
    
    # Fix tgt_mask shape for test
    # In train.py: tgt_mask = tgt_mask[:, :-1, :-1]
    # If tgt_mask was (B, 1, S, S)
    
    # Let's recreate masks exactly as in dataset.py
    import numpy as np
    def subsequent_mask(size):
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0
        
    tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2) # (B, 1, 1, S)
    tgt_sub_mask = subsequent_mask(tgt.size(-1)).type_as(tgt_pad_mask) # (1, S, S)
    tgt_mask = tgt_pad_mask & tgt_sub_mask # (B, 1, S, S)
    
    decoder_input = tgt[:, :-1]
    tgt_mask = tgt_mask[:, :, :-1, :-1]
    
    print(f"src shape: {src.shape}")
    print(f"decoder_input shape: {decoder_input.shape}")
    print(f"src_mask shape: {src_mask.shape}")
    print(f"tgt_mask shape: {tgt_mask.shape}")
    
    out = model.forward(src, decoder_input, src_mask, tgt_mask)
    print("Forward pass successful!")
    print(f"Output shape: {out.shape}")

if __name__ == "__main__":
    test_shapes()
