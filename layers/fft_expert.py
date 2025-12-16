# layers/fft_expert.py
import torch
import torch.nn as nn
import numpy as np


class FFT_Expert(nn.Module):

    def __init__(self, configs):
        super(FFT_Expert, self).__init__()
        self.d_model = configs.d_model
        self.top_k = getattr(configs, 'fft_top_k', 2)
        self.freq_processor = nn.Linear(self.d_model * 2, self.d_model * 2, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B_N, P, D = x.shape
        original_dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            x_float32 = x.to(torch.float32)
            xf = torch.fft.rfft(x_float32, dim=1)
            amps = torch.abs(xf).mean(dim=[0, 2])
            non_dc_amps = amps[1:]
            actual_k = min(self.top_k, len(non_dc_amps))

            if actual_k > 0:
                _, top_k_relative_indices = torch.topk(non_dc_amps, actual_k)
                top_k_indices = top_k_relative_indices + 1
            else:
                top_k_indices = torch.tensor([1], device=x.device) if len(amps) > 1 else torch.tensor([],
                                                                                                      device=x.device,
                                                                                                      dtype=torch.long)
            xf_filtered = torch.zeros_like(xf)
            if top_k_indices.numel() > 0:
                xf_filtered[:, top_k_indices, :] = xf[:, top_k_indices, :]

            xf_real = xf_filtered.real
            xf_imag = xf_filtered.imag
            xf_re_im = torch.cat([xf_real, xf_imag], dim=-1)

            processed_re_im = self.freq_processor(xf_re_im)

            processed_real, processed_imag = torch.split(processed_re_im, self.d_model, dim=-1)
            processed_xf = torch.complex(processed_real, processed_imag)

            x_out = torch.fft.irfft(processed_xf, n=P, dim=1)

        return x_out.to(original_dtype)