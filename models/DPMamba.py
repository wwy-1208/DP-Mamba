
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import numpy as np

from layers.RevIN import RevIN
from layers.Embed import PositionalEmbedding
from layers.MTST_Backbone import Flatten_Head
from layers.mamba_encoder import MambaBranchEncoder
from layers.fft_expert import FFT_Expert


class DynamicPatchFFTExpert(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.d_model = config.d_model
        self.patch_configs = [
            {'len': 12, 'stride': 6},
            {'len': 24, 'stride': 12},
            {'len': 48, 'stride': 24},
        ]
        self.num_experts = len(self.patch_configs)

        self.sub_experts = nn.ModuleList()
        for pc in self.patch_configs:
            patch_len = pc['len']
            stride = pc['stride']
            q_len = int((config.seq_len - patch_len) / stride + 1)

            sub_expert_modules = nn.ModuleDict({
                'patch_embed': nn.Linear(patch_len, self.d_model),
                'pos_embed': PositionalEmbedding(d_model=self.d_model, max_len=q_len),
                'fft_encoder': FFT_Expert(config)
            }).to(device)
            self.sub_experts.append(sub_expert_modules)
        self.top_k_freq = 3
        self.gate_network = nn.Sequential(
            nn.Linear(self.top_k_freq, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_experts)
        ).to(device)

        print(f"[DynamicPatchFFTExpert] Initialized with {self.num_experts} parallel FFT sub-experts.")

    def forward(self, x_coeff):  # x_coeff: [B, N, L]
        B, N, L = x_coeff.shape

        x_fft = torch.fft.rfft(x_coeff.to(torch.float32), dim=-1)
        amps = torch.abs(x_fft).mean(dim=[0, 1])

        non_dc_amps = amps[1:]
        top_k_actual = min(self.top_k_freq, len(non_dc_amps))
        if top_k_actual > 0:
            _, top_k_indices = torch.topk(non_dc_amps, top_k_actual)
            top_k_periods = L / (top_k_indices + 1)
        else:
            top_k_periods = torch.zeros(self.top_k_freq, device=self.device)

        if top_k_periods.numel() < self.top_k_freq:
            padding = torch.zeros(self.top_k_freq - top_k_periods.numel(), device=self.device)
            top_k_periods = torch.cat([top_k_periods, padding])

        gate_logits = self.gate_network(top_k_periods.detach())
        gate_weights = F.softmax(gate_logits, dim=-1)  # [num_experts]

        all_expert_outputs = []
        for i, expert_modules in enumerate(self.sub_experts):
            pc = self.patch_configs[i]
            patches = x_coeff.unfold(dimension=-1, size=pc['len'], step=pc['stride'])
            embedded_patches = expert_modules['patch_embed'](patches)
            u_branch = rearrange(embedded_patches, 'b n p d -> (b n) p d')
            u_branch = u_branch + expert_modules['pos_embed'](u_branch)

            fft_out = expert_modules['fft_encoder'](u_branch)

            target_len = self.sub_experts[0].pos_embed.pe.shape[1]
            fft_out_reshaped = F.interpolate(fft_out.transpose(1, 2), size=target_len, mode='linear').transpose(1, 2)
            all_expert_outputs.append(fft_out_reshaped)

        final_output = torch.zeros_like(all_expert_outputs[0])
        for i in range(self.num_experts):
            final_output += gate_weights[i] * all_expert_outputs[i]

        return final_output


class FusionModule(nn.Module):

    def __init__(self, d_model, config):
        super().__init__()
        self.ablation_mode = config.ablation_mode
        self.d_model = d_model

        if self.ablation_mode == 'moe':
            self.gate_linear = nn.Linear(d_model * 2, d_model)

        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        print(f"[FusionModule] Initialized in '{self.ablation_mode}' mode.")

    def forward(self, mamba_out, fft_out):
        if self.ablation_mode == 'moe':
            combined = torch.cat([mamba_out, fft_out], dim=-1)
            gate = torch.sigmoid(self.gate_linear(combined))
            fused_features = gate * mamba_out + (1 - gate) * fft_out
        elif self.ablation_mode == 'sum_experts':
            fused_features = mamba_out + fft_out
        elif self.ablation_mode == 's_mamba_only':
            return self.dropout(self.output_linear(mamba_out))
        elif self.ablation_mode == 'fft_only':
            return self.dropout(self.output_linear(fft_out))
        else:
            raise ValueError(f"Unknown ablation mode in FusionModule: {self.ablation_mode}")

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
            self.mamba_encoder = MambaBranchEncoder(config).to(device)
            print("  --> Mamba expert initialized.")
        if self.ablation_mode in ['moe', 'fft_only', 'sum_experts']:
            self.fft_encoder = FFT_Expert(config).to(device)
            print("  --> FFT expert initialized.")

        self.fusion = FusionModule(self.d_model, config).to(device)
        self.norm = nn.LayerNorm(self.d_model).to(device)

    def forward(self, x_coeff, B):
        patches = x_coeff.unfold(dimension=-1, size=self.patch_embed.in_features, step=self.stride)
        embedded_patches = self.patch_embed(patches)
        u_branch = rearrange(embedded_patches, 'b n p d -> (b n) p d')
        u_branch = self.dropout(u_branch + self.pos_embed(u_branch))

        mamba_out = self.mamba_encoder(u_branch) if hasattr(self, 'mamba_encoder') else None
        fft_out = self.fft_encoder(u_branch) if hasattr(self, 'fft_encoder') else None

        if self.ablation_mode == 's_mamba_only':
            fft_out = mamba_out
        elif self.ablation_mode == 'fft_only':
            mamba_out = fft_out

        fused_out = self.fusion(mamba_out, fft_out)

        if self.ablation_mode in ['moe', 'sum_experts']:
            output = self.norm(u_branch + fused_out)
        else:
            output = fused_out

        return output


class DPMambaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = getattr(config, 'device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.revin = RevIN(config.c_out, eps=1e-5, affine=config.affine).to(self.device) if config.revin else None

        patch_len = config.patch_len
        stride = config.stride
        input_len = config.seq_len
        q_len = int((input_len - patch_len) / stride + 1)

        if q_len <= 0:
            raise ValueError("分块参数设置错误 (patch_num <= 0)，请检查 seq_len, patch_len, 和 stride。")

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