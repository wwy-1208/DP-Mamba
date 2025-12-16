# layers/MTST_MoE_Encoder.py
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, List
from einops import rearrange
from layers.Embed import PositionalEmbedding
from layers.Long_encoder import S_Mamba_Expert
from layers.fft_expert import FFT_Expert


class MoELayerBranch(nn.Module):

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.d_model = config.d_model
        self.ablation_mode = getattr(config, 'ablation_mode', 'moe')

        self.last_gate_weights = None

        if self.ablation_mode in ["moe", "s_mamba_only", "sum_experts"]:
            class TempConfigSST:
                pass

            s_mamba_config_expert = TempConfigSST()
            for k, v in vars(config).items(): setattr(s_mamba_config_expert, k, v)
            s_mamba_config_expert.e_layers = 1
            self.s_mamba_expert = S_Mamba_Expert(s_mamba_config_expert).to(device)

        if self.ablation_mode in ["moe", "fft_only", "sum_experts"]:
            self.fft_expert = FFT_Expert(config).to(device)

        if self.ablation_mode == "moe":
            self.num_experts = 2
            self.gate = nn.Linear(self.d_model, self.num_experts, bias=False).to(device)
        else:
            self.gate = None

    def forward(self, x: Tensor, pos_emb: Optional[Tensor] = None) -> Tensor:
        B_N, P, C = x.shape

        if self.ablation_mode == "s_mamba_only":
            if not hasattr(self, 's_mamba_expert'):
                raise RuntimeError("消融模式为 's_mamba_only'，但 S_Mamba 專家未初始化。")
            return self.s_mamba_expert(x)

        elif self.ablation_mode == "fft_only":
            if not hasattr(self, 'fft_expert'):
                raise RuntimeError("消融模式为 'fft_only'，但 FFT 專家未初始化。")
            return self.fft_expert(x)

        elif self.ablation_mode == "sum_experts":
            if not all(hasattr(self, name) for name in ['s_mamba_expert', 'fft_expert']):
                raise RuntimeError("消融模式为 'sum_experts'，但所需的專家模塊未完全初始化。")
            mamba_output = self.s_mamba_expert(x)
            fft_output = self.fft_expert(x)
            return mamba_output + fft_output

        elif self.ablation_mode == "moe":
            if not all(hasattr(self, name) for name in
                       ['s_mamba_expert', 'fft_expert']) or self.gate is None:
                raise RuntimeError("消融模式为 'moe'，但所需的模塊（專家或門控網絡）未完全初始化。")

            context = x[:, 0, :]
            gate_logits = self.gate(context)
            gate_weights = F.softmax(gate_logits, dim=-1)

            self.last_gate_weights = gate_weights.detach()

            mamba_output = self.s_mamba_expert(x)
            fft_output = self.fft_expert(x)

            g_mamba = gate_weights[:, 0].unsqueeze(1).unsqueeze(2)
            g_fft = gate_weights[:, 1].unsqueeze(1).unsqueeze(2)

            weighted_output = g_mamba * mamba_output + g_fft * fft_output
            return weighted_output
        else:

            raise ValueError(f"未知的消融模式: {self.ablation_mode}。可用模式: 'moe', 's_mamba_only', 'fft_only', 'sum_experts'")

class BranchEncoder(nn.Module):
    def __init__(self, config, patch_len, stride, q_len, device):
        super().__init__()
        # ... (保留 config, patch_len, stride, q_len, d_model, n_layers, device) ...
        self.config = config
        self.patch_len = patch_len
        self.stride = stride
        self.q_len = q_len
        self.d_model = config.d_model
        self.n_layers = config.e_layers
        self.device = device
        self.padding_patch = config.padding_patch

        if self.padding_patch == 'end':
            # Padding on time dimension (last dim of [B, N, L])
            self.padding_layer = nn.ReplicationPad1d((0, self.stride))

        # Patch Projection (作用在 patch_len 维度)
        self.projection = nn.Linear(self.patch_len, self.d_model).to(device)

        # Positional Encoding (长度 q_len)
        self.pos_embed = PositionalEmbedding(self.d_model, max_len=self.q_len).to(device) # 或可学习 PE

        # MoE Layers + FF (保持不变)
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
             moe_block = MoELayerBranch(config, device=device)
             ff_block = nn.Sequential(
                 nn.Linear(self.d_model, config.d_ff),
                 nn.GELU() if config.activation == 'gelu' else nn.ReLU(),
                 nn.Dropout(config.dropout),
                 nn.Linear(config.d_ff, self.d_model),
                 nn.Dropout(config.dropout)
             ).to(device)
             norm1 = nn.LayerNorm(self.d_model).to(device)
             norm2 = nn.LayerNorm(self.d_model).to(device)
             self.layers.append(nn.ModuleList([norm1, moe_block, norm2, ff_block]))

    def forward(self, x_branch):
        # x_branch: [B, N, L] (N=C_in, L=seq_len)
        B, N, L = x_branch.shape

        # 1. Padding (if needed)
        if self.padding_patch == 'end':
            # Pad the L dimension
            # Need to reshape for Conv1d padding: [B*N, 1, L] or [B*N, L]?
            # ReplicationPad1d expects [B, C, L] or [C, L]
            # Reshape to [B*N, 1, L] might work, or pad manually / use F.pad
            # Using F.pad for simplicity: pads the last dimension
            x_padded = F.pad(x_branch, (0, self.stride), mode='replicate') # [B, N, L_padded]
        else:
            x_padded = x_branch

        # 2. Patching (Unfold on L dimension)
        # unfold output [B, N, num_patches, patch_len]
        patches = x_padded.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        if patches.shape[2] != self.q_len: # Check patch count
             print(f"警告: BranchEncoder patch 数量 ({patches.shape[2]}) 与预期 q_len ({self.q_len}) 不匹配。")
             # Handle mismatch (e.g., truncate/pad - simple truncation here)
             patches = patches[:, :, :self.q_len, :]
             if patches.shape[2] != self.q_len: raise ValueError("输入序列过短。")

        # 3. Projection
        # Input [B, N, q_len, patch_len] -> Output [B, N, q_len, d_model]
        projected_patches = self.projection(patches)

        # 4. Reshape for MoE Layers
        # Output [B*N, q_len, d_model]
        u = rearrange(projected_patches, 'b n p d -> (b n) p d')

        # 5. Apply MoE + FF Layers
        pos_emb_val = self.pos_embed(u) # Get PE of length q_len
        for norm1, moe_block, norm2, ff_block in self.layers:
            res_moe = u
            u_norm1 = norm1(u)
            moe_out = moe_block(u_norm1, pos_emb=pos_emb_val)
            u = res_moe + moe_out
            res_ff = u
            u_norm2 = norm2(u)
            ff_out = ff_block(u_norm2)
            u = res_ff + ff_out

        # 6. Reshape back to [B, N, q_len, d_model]
        output = rearrange(u, '(b n) p d -> b n p d', b=B)

        # 7. Permute to [B, N, d_model, q_len] for flattening in MultiBranchMoEEncoder
        output = output.permute(0, 1, 3, 2)

        return output # [B, N, d_model, q_len]


class MultiBranchMoEEncoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.n_branches = config.n_branches
        self.d_model = config.d_model

        self.patch_len_ls = [int(p) for p in config.patch_len_ls.split(',')]
        self.stride_ls = [int(s) for s in config.stride_ls.split(',')]
        assert len(self.patch_len_ls) == self.n_branches
        assert len(self.stride_ls) == self.n_branches

        self.patch_nums = []
        for i in range(self.n_branches):
            L = config.seq_len
            P = self.patch_len_ls[i]
            S = self.stride_ls[i]
            if config.padding_patch == 'end':

                L_padded = L + S # 简化假设，严格应为 L + (S - (L-P)%S)%S if (L-P)%S!=0 else L
                q_len = int((L_padded - P) / S + 1)
            else:
                q_len = int((L - P) / S + 1)
            self.patch_nums.append(q_len)

        self.branches = nn.ModuleList([
            BranchEncoder(
                config=config,
                patch_len=self.patch_len_ls[j],
                stride=self.stride_ls[j],
                q_len=self.patch_nums[j],
                device=device
            ) for j in range(self.n_branches)
        ])

        self.total_feature_dim = self.d_model * sum(self.patch_nums)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [B, N, L] (N=C_in, L=seq_len)
        branch_outputs_flattened = []

        for j in range(self.n_branches):
            # branch_out shape: [B, N, D, P_j]
            branch_out = self.branches[j](x)
            # Flatten D and P_j: [B, N, D * P_j]
            branch_flat = rearrange(branch_out, 'b n d p -> b n (d p)')
            branch_outputs_flattened.append(branch_flat)

        # Concatenate along the feature dimension (last dim)
        # combined_features shape: [B, N, total_feature_dim]
        combined_features = torch.cat(branch_outputs_flattened, dim=-1)

        return combined_features # [B, N, total_feature_dim]
