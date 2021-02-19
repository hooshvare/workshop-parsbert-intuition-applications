import torch
from torch._C import dtype
from torch.nn import functional as F
from torch import nn as nn
from positionals import (
    Embedding,
    FixedPositionalEncoding
)
from models import Encoder

import copy
import os
import numpy as np


class GELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Bert(nn.Module):

    def __init__(
        self,
        n_vocab: int,
        pad_idx: int,
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

        self.n_vocab = n_vocab
        self.pad_idx = pad_idx

        self.d_model = d_model
        self.heads = heads
        self.n_layers = n_layers
        self.n_position = n_position
        self.ff_d_ff = ff_d_ff
        self.ff_activation = ff_activation
        self.ff_bias1 = ff_bias1
        self.ff_bias2 = ff_bias2
        self.keep_attention = keep_attention

        # word token embedding
        self.wte = Embedding(n_vocab, d_model, pad_idx=pad_idx, scale_wte=scale_wte)

        # positional encoding
        self.pe = FixedPositionalEncoding(d_model, n_position, dropout_rate=dropout_rate)

        # segment position embedding
        self.spe = Embedding(2, d_model)

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

        self.fc = nn.Linear(d_model, d_model)
        self.activation_f1 = nn.Tanh()
        self.classifier = nn.Linear(d_model, 2)

        self.linear = nn.Linear(d_model, d_model)
        self.activation_f2 = GELU()
        self.norm = nn.LayerNorm(d_model)

        self.decoder = nn.Linear(d_model, n_vocab, bias=False)
        self.decoder.weight = self.wte.weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

        # Xavier weights initialization for all parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        masked_position: torch.Tensor,
    ):
        inputs = self.wte(input_ids) + self.spe(segment_ids)
        position_embedding = self.pe(inputs)

        attention_mask = self.make_attention_masking(input_ids, input_ids, self.pad_idx)

        if self.keep_attention:
            encoded, attention_list = self.encoder(position_embedding, attention_mask)
        else:
            encoded = self.encoder(position_embedding, attention_mask)

        pooling = self.activation_f1(encoded[:, 0])
        logits_classifier = self.classifier(pooling)

        masked_position = masked_position[:, :, None].expand(-1, -1, encoded.size(-1))
        masked = torch.gather(encoded, 1, masked_position)
        masked = self.norm(self.activation_f2(self.linear(masked)))
        logits_lm = self.decoder(masked) + self.decoder_bias

        if self.keep_attention:
            return logits_lm, logits_classifier, encoded, attention_list

        return logits_lm, logits_classifier, encoded

    def make_attention_masking(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        pad_idx: int = 0
    ):
        batch_size, query_len = query.size()
        batch_size, key_len = key.size()
        masking = key.data.eq(pad_idx).unsqueeze(1)
        masking = torch.as_tensor(masking, dtype=torch.int)
        return masking.expand(batch_size, query_len, key_len)
