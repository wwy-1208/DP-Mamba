# code from https://github.com/ts-kim/RevIN, with minor modifications

import torch
import torch.nn as nn
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):  # 移除device参数
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode:str):
        # 确保输入和参数在同一设备（依赖PyTorch自动管理）
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1).detach()
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(
                torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps  # 添加eps
            ).detach()
    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight.abs() + 1e-7)  # 防止权重接近零导致溢出
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        # 添加输出范围限制
        x = torch.clamp(x, min=-1e4, max=1e4)
        return x
