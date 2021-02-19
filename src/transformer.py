import torch
from torch.nn import functional as F
from torch import nn as nn
from positionals import (
    Embedding,
    FixedPositionalEncoding
)
from models import Encoder, Decoder

import copy
import os



class Transformer(nn.Module):

    def __init__(
        self,
        src_n_vocab: int,
        tgt_n_vocab: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        d_model: int,
        heads: int,
        ff_d_ff: int,
        n_layers: int,
        n_position: int = 5000,
        ff_activation: nn.Module = nn.ReLU(),
        ff_bias1: bool = True,
        ff_bias2: bool = True,
        dropout_rate: float = 0.1,
        scale_wte: bool = False,
        keep_attention: bool = False
    ):
        super().__init__()

        self.src_n_vocab = src_n_vocab
        self.tgt_n_vocab = tgt_n_vocab

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.d_model = d_model
        self.heads = heads
        self.n_layers = n_layers
        self.n_position = n_position
        self.ff_d_ff = ff_d_ff
        self.ff_activation = ff_activation
        self.ff_bias1 = ff_bias1
        self.ff_bias2 = ff_bias2
        self.keep_attention = keep_attention

        # Encoder
        self.src_wte = Embedding(src_n_vocab, d_model, pad_idx=src_pad_idx, scale_wte=scale_wte)
        self.src_pe = FixedPositionalEncoding(d_model, n_position, dropout_rate=dropout_rate)
        self.encoder = Encoder(
            d_model=d_model,
            heads=heads,
            n_layers=n_layers,
            ff_d_ff=ff_d_ff,
            ff_activation=ff_activation,
            ff_bias1=ff_bias1,
            ff_bias2=ff_bias2,
            dropout_rate=dropout_rate,
            keep_attention=keep_attention
        )

        # Decoder
        self.tgt_wte = Embedding(tgt_n_vocab, d_model, pad_idx=tgt_pad_idx, scale_wte=scale_wte)
        self.tgt_pe = FixedPositionalEncoding(d_model, n_position, dropout_rate=dropout_rate)
        self.decoder = Decoder(
            d_model=d_model,
            heads=heads,
            n_layers=n_layers,
            ff_d_ff=ff_d_ff,
            ff_activation=ff_activation,
            ff_bias1=ff_bias1,
            ff_bias2=ff_bias2,
            dropout_rate=dropout_rate,
            keep_attention=keep_attention
        )

        self.decoder_projection = nn.Linear(d_model, tgt_n_vocab)
        self.decoder_log_softmax = nn.LogSoftmax(dim=1)

        # Xavier weights initialization for all parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(
        self,
        src_input_ids: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        tgt_mask: torch.Tensor
    ):
        encoded = self.encoding(src_input_ids, src_mask)
        decoded_log_probs = self.decoding(tgt_input_ids, tgt_mask, encoded, src_mask)
        return decoded_log_probs

    def encoding(
        self,
        src_input_ids: torch.Tensor,
        src_mask: torch.Tensor,
    ):
        embedding = self.src_wte(src_input_ids)
        position_embedding = self.src_pe(embedding)

        if self.keep_attention:
            encoded, attention_list = self.encoder(position_embedding, src_mask)
            return encoded, attention_list

        encoded = self.encoder(position_embedding, src_mask)
        return encoded

    def decoding(
        self,
        tgt_input_ids: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_encoded: torch.Tensor,
        src_mask: torch.Tensor,
    ):
        embedding = self.src_wte(tgt_input_ids)
        position_embedding = self.src_pe(embedding)

        if self.keep_attention:
            decoded, attention_list = self.decoder(position_embedding, tgt_mask, src_encoded, src_mask)
        else:
            decoded = self.decoder(position_embedding, tgt_mask, src_encoded, src_mask)

        # shape = (batch_size, seq_len, n_vocab)
        log_probs = self.decoder_log_softmax(self.decoder_projection(decoded))

        # shape = (batch_size * seq_len, n_vocab) for label smoothing ...
        log_probs = log_probs.reshape(-1, log_probs.shape[-1])

        if self.keep_attention:
            return log_probs, attention_list

        return log_probs
