"""A real, recognizable transformer encoder block - not an arbitrary
academic shape. d_model=512, nhead=8, dim_feedforward=2048 is the
original "Transformer base" config (Vaswani et al. 2017), the same
shape family as an actual small model layer, not a stand-in.

Milestone 1: routing decisions on a real multi-op forward pass (self
-attention + MLP + layernorms + residuals), not one isolated matmul.
Backends call forward() with a real (batch, seq_len, d_model) tensor -
this exercises many real ops per call, closer to what routing an actual
model's prefill stage would look like.
"""
import torch
import torch.nn as nn

D_MODEL = 512
N_HEAD = 8
DIM_FEEDFORWARD = 2048


class TinyTransformerBlock(nn.Module):
    """One standard post-norm transformer encoder layer - the same
    building block real models stack N times. Deliberately just
    `nn.TransformerEncoderLayer` (PyTorch's own real implementation,
    not a hand-rolled stand-in) so the ops routed here are the exact
    ops a real model would run.
    """

    def __init__(self):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def make_input(batch: int, seq_len: int, device: str) -> torch.Tensor:
    return torch.randn(batch, seq_len, D_MODEL, device=device)
