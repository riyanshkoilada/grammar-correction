"""Training script for the Transformer model.

This script handles the training loop, including data loading, model initialization,
optimization, and checkpointing. It uses argparse for command-line configuration.
"""

import argparse
import logging
import os
import time
from typing import Iterator, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import T5Tokenizer

from .dataset import GrammarDataset, collate_batch
from .model import make_model, Transformer
from .optimizer import NoamOpt
from .utils import setup_logging, get_device


def save_checkpoint(
    path: str,
    model: Transformer,
    optimizer: NoamOpt,
    epoch: int,
    step: int,
    loss: float,
    logger: logging.Logger
) -> None:
    """Saves a training checkpoint.

    Args:
        path (str): Path to save the checkpoint.
        model (Transformer): The model.
        optimizer (NoamOpt): The optimizer wrapper.
        epoch (int): Current epoch.
        step (int): Current step.
        loss (float): Current loss.
        logger (logging.Logger): Logger instance.
    """
    state_dict = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.optimizer.state_dict(),
        'optimizer_step': optimizer._step, # Save NoamOpt internal step
        'loss': loss
    }
    torch.save(state_dict, path)
    logger.info(f"Saved checkpoint to {path}")

def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: NoamOpt,
    logger: logging.Logger
) -> Tuple[int, int]:
    """Loads a training checkpoint.

    Args:
        path (str): Path to the checkpoint file.
        model (Transformer): The model.
        optimizer (NoamOpt): The optimizer wrapper.
        logger (logging.Logger): Logger instance.

    Returns:
        Tuple[int, int]: The epoch and step to resume from.
    """
    logger.info(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer._step = checkpoint.get('optimizer_step', 0) # Restore NoamOpt step
    
    epoch = checkpoint['epoch']
    step = checkpoint.get('step', 0)
    
    logger.info(f"Resumed from epoch {epoch}, step {step}")
    return epoch, step


# Mixed Precision Scaler
scaler = torch.amp.GradScaler('cuda')

def run_epoch(
    data_iter: Iterator,
    model: Transformer,
    loss_compute: 'SimpleLossCompute',
    optimizer: Optional[NoamOpt],
    logger: logging.Logger,
    start_step: int = 0,
    epoch_idx: int = 0
) -> float:
    """Standard Training and Logging Function with AMP support."""
    start = time.time()
    total_tokens = 0
    total_loss = 0.0
    tokens = 0
    
    for i, batch in enumerate(data_iter):
        if i < start_step:
            continue
            
        src = batch['src'].cuda()
        tgt = batch['tgt'].cuda()
        src_mask = batch['src_mask'].cuda()
        tgt_mask = batch['tgt_mask'].cuda()
        
        decoder_input = tgt[:, :-1]
        target = tgt[:, 1:]
        tgt_mask = tgt_mask[:, :, :-1, :-1]
        
        # Mixed Precision Context
        with torch.amp.autocast('cuda'):
            out = model.forward(src, decoder_input, src_mask, tgt_mask)
            loss = loss_compute(out, target, batch['tgt_mask'][:, 1:, 1:])
        
        total_loss += loss
        total_tokens += (target != 0).sum().item() 
        
        if optimizer is not None:
            # Scaled Backward & Step
            scaler.scale(loss_compute.last_loss_tensor).backward()
            
            # Unscale before step (for NoamOpt which might inspect grads)
            scaler.unscale_(optimizer.optimizer)
            
            # Step with scaler
            # Note: NoamOpt wraps the optimizer step. We need to be careful.
            # NoamOpt does `self.optimizer.step()`.
            # Scaler expects `scaler.step(optimizer)`.
            # We will manually step scaler using the inner optimizer.
            
            scaler.step(optimizer.optimizer)
            scaler.update()
            
            # Update NoamOpt parameters (rate) manually since we bypassed its .step()
            optimizer._step += 1
            rate = optimizer.rate()
            for p in optimizer.optimizer.param_groups:
                p['lr'] = rate
            optimizer._rate = rate
            
            optimizer.zero_grad()
            
        if i % 10 == 0:
            elapsed = time.time() - start
            if elapsed == 0:
                elapsed = 0.001
            logger.info(
                "Epoch %d Step: %d Loss: %f Tokens per Sec: %f",
                epoch_idx, i, loss / batch['src'].size(0), total_tokens / elapsed
            )
            start = time.time()
            tokens = 0
            
        if i % 5000 == 0 and i > 0 and optimizer is not None:
             save_checkpoint(
                f"checkpoint_epoch_{epoch_idx}_step_{i}.pt",
                model, optimizer, epoch_idx, i, loss, logger
            )
            
    return total_loss / (total_tokens + 1e-9)

class SimpleLossCompute:
    """A simple loss compute and train function."""
    def __init__(self, generator: nn.Module, criterion: nn.Module):
        self.generator = generator
        self.criterion = criterion
        self.last_loss_tensor = None # To store tensor for backward
        
    def __call__(self, x: torch.Tensor, y: torch.Tensor, norm: torch.Tensor) -> float:
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1))
        
        self.last_loss_tensor = loss # Store for scaling outside
        return loss.item()

def train(args: argparse.Namespace) -> None:
    """Main training function."""
    logger = setup_logging()
    
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    vocab_size = tokenizer.vocab_size
    
    # Dataset
    logger.info("Loading dataset...")
    num_samples = args.num_samples if args.num_samples > 0 else None
    
    train_dataset = GrammarDataset(tokenizer, split='train', num_samples=num_samples)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=True if args.device != 'cpu' else False
    )
    
    # Model
    logger.info("Creating model...")
    model = make_model(
        vocab_size, 
        vocab_size, 
        N=args.n_layers, 
        d_model=args.d_model, 
        d_ff=args.d_ff, 
        h=args.heads
    )
    model.to(device)

    # Optional optimizations
    if args.tf32 and device.type == 'cuda':
        logger.info("Enabling TensorFloat-32 (TF32)")
        torch.set_float32_matmul_precision('high')
    
    if args.compile:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Optimizer
    optimizer = NoamOpt(
        args.d_model, 
        1, 
        args.warmup,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    )
    
    # Resume logic
    start_epoch = 0
    start_step = 0
    if args.resume_from:
        if os.path.exists(args.resume_from):
            start_epoch, start_step = load_checkpoint(args.resume_from, model, optimizer, logger)
        else:
            logger.warning(f"Checkpoint {args.resume_from} not found. Starting from scratch.")
    
    # Loss
    criterion = nn.NLLLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
    loss_compute = SimpleLossCompute(model.generator, criterion)

    # Training Loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        logger.info(f"Epoch {epoch} Training...")
        
        # If resuming within an epoch, pass start_step. Otherwise 0.
        current_start_step = start_step if epoch == start_epoch else 0
        
        run_epoch(train_loader, model, loss_compute, optimizer, logger, start_step=current_start_step, epoch_idx=epoch)
        
        # Save end of epoch checkpoint
        save_checkpoint(
            f"model_epoch_{epoch}.pt",
            model, optimizer, epoch + 1, 0, 0.0, logger
        )

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Transformer Model")
    
    # Data
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples to train on (0 for all)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    
    # Model Architecture
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers (N)")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension (d_model)")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    
    # Training
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--warmup", type=int, default=400, help="Warmup steps")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    
    # Resume
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")

    # Optimization
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--tf32", action="store_true", default=True, help="Enable TF32 (default: True)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
