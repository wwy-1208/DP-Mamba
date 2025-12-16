# layers/mamba_encoder.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from layers.Mamba_EnDec import Encoder, EncoderLayer

class MambaBranchEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
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
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        encoder_output, _ = self.encoder(x)
        return encoder_output