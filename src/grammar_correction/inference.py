import torch
from transformers import T5Tokenizer
from .model import make_model
import sys
import os

def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           ys, 
                           torch.tensor([1]).type_as(src.data)) # Dummy mask for now, or subsequent mask
        # Actually, decode expects tgt_mask.
        # For inference, we can just pass subsequent mask for current length
        # But our model.decode calls decoder which expects tgt_mask
        
        # Let's create proper mask
        sz = ys.size(1)
        tgt_mask = (torch.triu(torch.ones(1, sz, sz)) == 1).transpose(1, 2)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        # Wait, our model expects boolean mask where False is masked?
        # In model.py: scores.masked_fill(mask == 0, -1e9)
        # So mask should be 1 for valid, 0 for invalid.
        
        import numpy as np
        attn_shape = (1, sz, sz)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        tgt_mask = (torch.from_numpy(subsequent_mask) == 0).type_as(src.data)
        
        out = model.decode(memory, src_mask, ys, tgt_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == end_symbol:
            break
    return ys

def load_model(path, device):
    # Hyperparameters must match training
    D_MODEL = 512
    N_LAYERS = 6
    HEADS = 8
    D_FF = 2048
    
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    vocab_size = tokenizer.vocab_size
    
    model = make_model(vocab_size, vocab_size, N=N_LAYERS, d_model=D_MODEL, d_ff=D_FF, h=HEADS)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer

def correct_grammar(input_text, model, tokenizer, device):
    model.eval()
    src_tokens = tokenizer(input_text, return_tensors='pt')
    src = src_tokens.input_ids.to(device)
    src_mask = (src != tokenizer.pad_token_id).unsqueeze(-2).to(device)
    
    out_tokens = greedy_decode(model, src, src_mask, max_len=64, start_symbol=tokenizer.pad_token_id, end_symbol=tokenizer.eos_token_id)
    # T5 uses pad_token_id as start symbol usually? Or decoder_start_token_id.
    # T5Tokenizer: pad_token_id=0, eos_token_id=1.
    # We trained with tgt shifted.
    # If we assume standard T5 training, decoder input starts with pad (0).
    
    decoded_text = tokenizer.decode(out_tokens[0], skip_special_tokens=True)
    return decoded_text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <model_path> <sentence>")
        # Default test
        model_path = "model_epoch_0.pt" # Example
        input_text = "He go to school yesterday."
    else:
        model_path = sys.argv[1]
        input_text = sys.argv[2]
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train first.")
        sys.exit(1)
        
    print(f"Loading model from {model_path}...")
    model, tokenizer = load_model(model_path, device)
    
    print(f"Input: {input_text}")
    correction = correct_grammar(input_text, model, tokenizer, device)
    print(f"Correction: {correction}")
