import torch
import torch.nn as nn
from layers.Mamba_EnDec import Encoder, EncoderLayer
from mamba_ssm import Mamba

from typing import Optional, Tuple
class S_Mamba_Expert(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.d_model = configs.d_model
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=configs.d_model,
                        d_state=configs.d_state,
                        d_conv=configs.d_conv,
                        expand=1,
                    ),
                    Mamba(
                        d_model=configs.d_model,
                        d_state=configs.d_state,
                        d_conv=configs.d_conv,
                        expand=1,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model) if getattr(configs, 'norm_layer', True) else None
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        encoder_output, _ = self.encoder(x, attn_mask=mask)
        return encoder_output