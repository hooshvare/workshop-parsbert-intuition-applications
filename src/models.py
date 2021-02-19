import torch
from torch import nn as nn

from attentions import MultiHeadAttention
from layers import EncoderLayer, DecoderLayer
from positionals import PositionWiseFeedForward

import copy
import os


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        n_layers: int,
        ff_d_ff: int,
        ff_activation: nn.Module = nn.ReLU(),
        ff_bias1: bool = True,
        ff_bias2: bool = True,
        dropout_rate: float = 0.1,
        keep_attention: bool = False
    ):
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.ff_d_ff = ff_d_ff
        self.n_layers = n_layers
        self.ff_activation = ff_activation
        self.ff_bias1 = ff_bias1
        self.ff_bias2 = ff_bias2
        self.keep_attention = keep_attention

        attention_layer = MultiHeadAttention(
            d_model=d_model,
            heads=heads,
            dropout_rate=dropout_rate,
            keep_attention=keep_attention
        )
        ff_layer = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=ff_d_ff,
            dropout_rate=dropout_rate,
            activation=ff_activation,
            bias1=ff_bias1,
            bias2=ff_bias2,
        )
        encoder_layer = EncoderLayer(
            d_model=d_model,
            attention=attention_layer,
            feed_forward=ff_layer,
            keep_attention=keep_attention
        )
        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_layer) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor
    ):
        # src shape: (batch_size, seq_len, d_model)
        # src_mask shape: (batch_size, 1, 1, seq_len)
        attention_list_layers = []

        x = src
        for layer in self.layers:
            if self.keep_attention:
                x, attention_list = layer(src=x, src_mask=src_mask)
                attention_list_layers.extend([attention_list])
            else:
                x = layer(src=x, src_mask=src_mask)

        x = self.norm(x)

        if self.keep_attention:
            return x, attention_list_layers

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        n_layers: int,
        ff_d_ff: int,
        ff_activation: nn.Module = nn.ReLU(),
        ff_bias1: bool = True,
        ff_bias2: bool = True,
        dropout_rate: float = 0.1,
        keep_attention: bool = False
    ):
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.ff_d_ff = ff_d_ff
        self.n_layers = n_layers
        self.ff_activation = ff_activation
        self.ff_bias1 = ff_bias1
        self.ff_bias2 = ff_bias2
        self.keep_attention = keep_attention

        attention_layer = MultiHeadAttention(
            d_model=d_model,
            heads=heads,
            dropout_rate=dropout_rate,
            keep_attention=keep_attention
        )
        src_attention_layer = MultiHeadAttention(
            d_model=d_model,
            heads=heads,
            dropout_rate=dropout_rate,
            keep_attention=keep_attention
        )
        ff_layer = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=ff_d_ff,
            dropout_rate=dropout_rate,
            activation=ff_activation,
            bias1=ff_bias1,
            bias2=ff_bias2,
        )
        decoder_layer = DecoderLayer(
            d_model=d_model,
            attention=attention_layer,
            src_attention=src_attention_layer,
            feed_forward=ff_layer,
            keep_attention=keep_attention
        )

        self.layers = nn.ModuleList([
            copy.deepcopy(decoder_layer) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        src: torch.Tensor,
        src_mask: torch.Tensor
    ):
        # tgt shape: (batch_size, seq_len, d_model)
        # tgt_mask shape: (batch_size, 1, seq_len, seq_len)

        # src shape: (batch_size, seq_len, d_model)
        # src_mask shape: (batch_size, 1, 1, seq_len)

        attention_list_layers = []

        x = tgt
        for layer in self.layers:
            if self.keep_attention:
                x, attention_list = layer(tgt=x, tgt_mask=tgt_mask, src=src, src_mask=src_mask)
                attention_list_layers.extend([attention_list])
            else:
                x = layer(tgt=x, tgt_mask=tgt_mask, src=src, src_mask=src_mask)

        x = self.norm(x)

        if self.keep_attention:
            return x, attention_list_layers

        return x
