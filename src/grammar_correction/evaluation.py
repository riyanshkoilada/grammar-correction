import torch
from transformers import T5Tokenizer
from .model import make_model
from .dataset import GrammarDataset
from torch.utils.data import DataLoader
from .inference import greedy_decode
import sacrebleu
# from gleu import GLEU
import sys
import os

def evaluate_model(model_path, device):
    # Hyperparameters must match training
    D_MODEL = 512
    N_LAYERS = 6
    HEADS = 8
    D_FF = 2048
    
    print("Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    vocab_size = tokenizer.vocab_size
    
    print(f"Loading model from {model_path}...")
    model = make_model(vocab_size, vocab_size, N=N_LAYERS, d_model=D_MODEL, d_ff=D_FF, h=HEADS)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("Loading validation dataset (JFLEG)...")
    val_dataset = GrammarDataset(tokenizer, split='validation')
    # Use a small subset for quick evaluation if needed, but JFLEG is small (747 sentences)
    
    references = []
    hypotheses = []
    inputs = []
    
    print("Running inference on validation set...")
    # GLEU requires list of references for each hypothesis
    # JFLEG provides multiple corrections per sentence.
    # Our dataset loader currently picks the first correction as 'output'.
    # For proper evaluation, we should load all corrections.
    # But let's stick to 1-to-1 for now for simplicity or modify dataset.py if needed.
    # Actually, let's just use the 'output' from dataset as single reference.
    
    for i, item in enumerate(val_dataset.data):
        src_text = item['input']
        tgt_text = item['output']
        
        src_tokens = tokenizer(src_text, return_tensors='pt')
        src = src_tokens.input_ids.to(device)
        src_mask = (src != tokenizer.pad_token_id).unsqueeze(-2).to(device)
        
        with torch.no_grad():
            out_tokens = greedy_decode(model, src, src_mask, max_len=64, start_symbol=tokenizer.pad_token_id, end_symbol=tokenizer.eos_token_id)
            
        pred_text = tokenizer.decode(out_tokens[0], skip_special_tokens=True)
        
        inputs.append(src_text)
        references.append([tgt_text]) # List of references for this sentence
        hypotheses.append(pred_text)
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(val_dataset)} samples")
            
    # Calculate BLEU
    # sacrebleu expects list of hypotheses and list of reference lists (transposed)
    # references is currently [[ref1], [ref2], ...]
    # sacrebleu wants [[ref1_doc1, ref1_doc2...], [ref2_doc1, ref2_doc2...]]
    # Since we have 1 ref per doc:
    refs_transposed = [[r[0] for r in references]]
    
    bleu = sacrebleu.corpus_bleu(hypotheses, refs_transposed)
    print(f"BLEU Score: {bleu.score}")
    
    # Calculate Exact Match
    exact_matches = sum([1 for h, r in zip(hypotheses, references) if h == r[0]])
    accuracy = exact_matches / len(hypotheses)
    print(f"Exact Match Accuracy: {accuracy:.2%}")
    
    # Generate Report
    report_path = "validation_report.md"
    with open(report_path, "w") as f:
        f.write("# Model Validation Report\n\n")
        f.write(f"**Model:** {model_path}\n")
        f.write(f"**BLEU Score:** {bleu.score:.2f}\n")
        f.write(f"**Exact Match Accuracy:** {accuracy:.2%}\n\n")
        f.write("## Sample Predictions\n\n")
        f.write("| Input | Prediction | Reference |\n")
        f.write("|-------|------------|-----------|\n")
        
        for i in range(min(20, len(inputs))):
            f.write(f"| {inputs[i]} | {hypotheses[i]} | {references[i][0]} |\n")
            
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Find latest model
    import glob
    models = glob.glob("model_epoch_*.pt")
    if not models:
        print("No models found.")
        sys.exit(1)
    latest_model = max(models, key=os.path.getctime)
    
    evaluate_model(latest_model, device)
