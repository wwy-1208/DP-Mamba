

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

from layers.RevIN import RevIN
from layers.Embed import PositionalEmbedding
from layers.MTST_Backbone import Flatten_Head
from layers.fft_expert import FFT_Expert
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerBranchEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=config.e_layers)

    def forward(self, x):
        return self.encoder(x)

class FusionModule(nn.Module):
    def __init__(self, d_model, config):
        super().__init__()
        self.ablation_mode = config.ablation_mode
        self.d_model = d_model
        if self.ablation_mode == 'moe':
            self.gate_linear = nn.Linear(d_model * 2, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, transformer_out, fft_out):
        if self.ablation_mode == 'moe':
            combined = torch.cat([transformer_out, fft_out], dim=-1)
            gate = torch.sigmoid(self.gate_linear(combined))
            fused_features = gate * transformer_out + (1 - gate) * fft_out
        elif self.ablation_mode == 'sum_experts':
            fused_features = transformer_out + fft_out
        elif self.ablation_mode == 's_mamba_only':
            return self.dropout(self.output_linear(transformer_out))
        elif self.ablation_mode == 'fft_only':
            return self.dropout(self.output_linear(fft_out))
        else:
            raise ValueError(f"Unknown ablation mode: {self.ablation_mode}")
        return self.dropout(self.output_linear(fused_features))


class ParallelBranch(nn.Module):
    def __init__(self, config, patch_len, stride, q_len, device):
        super().__init__()
        self.d_model = config.d_model
        self.stride = stride
        self.ablation_mode = config.ablation_mode
        self.patch_embed = nn.Linear(patch_len, self.d_model).to(device)
        self.pos_embed = PositionalEmbedding(d_model=self.d_model, max_len=q_len).to(device)
        self.dropout = nn.Dropout(config.dropout)

        if self.ablation_mode in ['moe', 's_mamba_only', 'sum_experts']:
            self.transformer_encoder = TransformerBranchEncoder(config).to(device)
        if self.ablation_mode in ['moe', 'fft_only', 'sum_experts']:
            self.fft_encoder = FFT_Expert(config).to(device)

        self.fusion = FusionModule(self.d_model, config).to(device)
        self.norm = nn.LayerNorm(self.d_model).to(device)

    def forward(self, x_coeff, B):
        patches = x_coeff.unfold(dimension=-1, size=self.patch_embed.in_features, step=self.stride)
        embedded_patches = self.patch_embed(patches)
        u_branch = rearrange(embedded_patches, 'b n p d -> (b n) p d')
        u_branch = self.dropout(u_branch + self.pos_embed(u_branch))

        transformer_out = self.transformer_encoder(u_branch) if hasattr(self, 'transformer_encoder') else None
        fft_out = self.fft_encoder(u_branch) if hasattr(self, 'fft_encoder') else None

        if self.ablation_mode == 's_mamba_only':
            fft_out = transformer_out
        elif self.ablation_mode == 'fft_only':
            transformer_out = fft_out

        fused_out = self.fusion(transformer_out, fft_out)

        if self.ablation_mode in ['moe', 'sum_experts']:
            output = self.norm(u_branch + fused_out)
        else:
            output = fused_out
        return output


class DP_Transformer_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = getattr(config, 'device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"SST_Transformer_Model Init: device={self.device}, mode={config.ablation_mode}")
        self.revin = RevIN(config.c_out, eps=1e-5, affine=config.affine).to(self.device) if config.revin else None
        patch_len = config.patch_len
        stride = config.stride
        input_len = config.seq_len
        q_len = int((input_len - patch_len) / stride + 1)
        if q_len <= 0:
            raise ValueError("分块参数设置错误。")
        self.processing_branch = ParallelBranch(config, patch_len, stride, q_len, self.device)
        total_output_dim = q_len * config.d_model
        self.head = Flatten_Head(
            individual=config.individual, n_vars=config.c_out,
            nf=total_output_dim, target_window=config.pred_len,
            head_dropout=getattr(config, 'head_dropout', 0.0)
        ).to(self.device)

    def forward(self, x, x_mark=None):
        B, L, N = x.shape
        x_norm = self.revin(x, 'norm') if self.revin else x
        x_permuted = x_norm.permute(0, 2, 1)
        processed_coeffs = self.processing_branch(x_permuted, B)
        branch_flat = rearrange(processed_coeffs, '(b n) p d -> b n (p d)', b=B)
        dec_out = self.head(branch_flat)
        dec_out = dec_out.permute(0, 2, 1)
        if self.revin:
            dec_out = self.revin(dec_out, 'denorm')
        return dec_out[:, :, :self.config.c_out]
