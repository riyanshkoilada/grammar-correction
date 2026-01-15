"""Transformer Model Implementation.

This module implements the complete Transformer architecture as described in
"Attention Is All You Need". It includes the Encoder, Decoder, Attention mechanisms,
and helper classes.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable

class Embeddings(nn.Module):
    """Standard embeddings with scaling."""
    def __init__(self, d_model: int, vocab: int):
        """Initializes the Embeddings layer.

        Args:
            d_model (int): The dimension of the model/embeddings.
            vocab (int): The size of the vocabulary.
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes embeddings multiplied by sqrt(d_model).

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len).

        Returns:
            torch.Tensor: Embedding tensor of shape (batch, seq_len, d_model).
        """
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens."""
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        """Initializes PositionalEncoding.

        Args:
            d_model (int): The dimension of the model.
            dropout (float): Dropout probability.
            max_len (int): Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        # 10000^(2i/d_model) term
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with positional encodings added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Produces N identical layers.

    Args:
        module (nn.Module): The module to clone.
        N (int): Number of copies.

    Returns:
        nn.ModuleList: List of cloned modules.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    """Construct a layernorm module."""
    def __init__(self, features: int, eps: float = 1e-6):
        """Initializes LayerNorm.

        Args:
            features (int): Number of features in the input.
            eps (float): Epsilon for numerical stability.
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Layer Normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """A residual connection followed by a layer norm."""
    def __init__(self, size: int, dropout: float):
        """Initializes SublayerConnection.

        Args:
            size (int): The feature size.
            dropout (float): Dropout probability.
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Apply residual connection to any sublayer with the same size.

        Args:
            x (torch.Tensor): Input tensor.
            sublayer (Callable): The sublayer function to apply.

        Returns:
            torch.Tensor: Output tensor.
        """
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward."""
    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module, dropout: float):
        """Initializes EncoderLayer.

        Args:
            size (int): The feature size.
            self_attn (nn.Module): Self-attention module.
            feed_forward (nn.Module): Feed-forward module.
            dropout (float): Dropout probability.
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pass the input through the encoder layer.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    """Core encoder is a stack of N layers."""
    def __init__(self, layer: EncoderLayer, N: int):
        """Initializes Encoder.

        Args:
            layer (EncoderLayer): The layer to stack.
            N (int): Number of layers.
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pass the input through each layer in turn.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Encoded output.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward."""
    def __init__(self, size: int, self_attn: nn.Module, src_attn: nn.Module, feed_forward: nn.Module, dropout: float):
        """Initializes DecoderLayer.

        Args:
            size (int): Feature size.
            self_attn (nn.Module): Self-attention module.
            src_attn (nn.Module): Source-attention module.
            feed_forward (nn.Module): Feed-forward module.
            dropout (float): Dropout probability.
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """Pass input through decoder layer options."""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    """Generic N layer decoder with masking."""
    def __init__(self, layer: DecoderLayer, N: int):
        """Initializes Decoder.

        Args:
            layer (DecoderLayer): Layer to stack.
            N (int): Number of layers.
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """Pass input through all decoder layers."""
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None, dropout: Optional[nn.Dropout] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 'Scaled Dot Product Attention'."""
    """Compute 'Scaled Dot Product Attention' using PyTorch 2.0 SDPA for Flash Attention."""
    # SDPA expects query, key, value to be (batch, heads, seq_len, head_dim)
    # The caller manages dimensions.
    
    # Handle dropout for SDPA
    dropout_p = 0.0
    if dropout is not None:
        dropout_p = dropout.p

    # mask in current codebase is 1 for keep, 0 for discard.
    # SDPA supports boolean mask where True = keep.
    attn_mask = mask
    if attn_mask is not None:
        # Convert 1/0 mask to boolean if it isn't already
        if attn_mask.dtype != torch.bool:
             attn_mask = attn_mask.bool()

    # Efficient implementation using Flash Attention if available (CUDA)
    # or efficient C++ implementation on CPU.
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p
    ), None

class MultiHeadAttention(nn.Module):
    """Implements Multi-Head Attention."""
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        """Initializes MultiHeadAttention.

        Args:
            h (int): Number of heads.
            d_model (int): Model dimension.
            dropout (float): Dropout probability.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Implements attention forward pass."""
        if mask is not None:
            # Same mask applied to all h heads.
            pass
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """Initializes PositionwiseFeedForward.

        Args:
            d_model (int): Model dimension.
            d_ff (int): Hidden layer size.
            dropout (float): Dropout probability.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements FFN forward pass."""
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, d_model: int, vocab: int):
        """Initializes Generator.

        Args:
            d_model (int): Model dimension.
            vocab (int): Vocabulary size.
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generates log-probabilities."""
        return F.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    """A standard Encoder-Decoder architecture."""
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: nn.Module, tgt_embed: nn.Module, generator: Generator):
        """Initializes Transformer."""
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """Take in and process masked src and target sequences."""
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """Encodes the source sequence."""
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """Decodes the target sequence."""
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

def make_model(src_vocab: int, tgt_vocab: int, N: int = 6, d_model: int = 512, d_ff: int = 2048, h: int = 8, dropout: float = 0.1) -> Transformer:
    """Helper: Construct a model from hyperparameters.

    Args:
        src_vocab (int): Source vocabulary size.
        tgt_vocab (int): Target vocabulary size.
        N (int): Number of layers.
        d_model (int): Model dimension.
        d_ff (int): Feed-forward hidden dimension.
        h (int): Number of attention heads.
        dropout (float): Dropout probability.

    Returns:
        Transformer: The constructed model.
    """
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
