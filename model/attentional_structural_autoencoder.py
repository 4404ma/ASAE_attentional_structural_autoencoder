import torch
from torch import nn
from .attention_encoder import AttentionEncoder
from .static_feature import StaticFeature
from .structural_decoder import Decoder


class ASAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention_encoder = AttentionEncoder(image_size=32, batch_size=100)
        self.static_feature = StaticFeature()
        self.structural_decoder = Decoder(batch_size=100)

    def forward(self, x):
        q = self.attention_encoder(x)
        x = self.static_feature(x)
        x = self.structural_decoder(x, q)

        return x