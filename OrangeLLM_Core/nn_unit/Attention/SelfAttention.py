import torch
import torch.nn as nn
from Auto_Config.base import AutoCfgModule
from Auto_Config.global_config_reader import global_cfg

nn.MultiheadAttention

class SelfAttention_by_handcode(nn.Module):

    def __init__(self, hidden_dim: int) -> None:
        """初始化线性层，创建权重参数，并按需创建偏置参数。

        Args:
            num_in_features: 输入特征数量。
            num_out_features: 输出特征数量。
            enable_bias: 是否启用偏置。
        """
        super(Linear_by_handcode, self).__init__()
        self.num_in_features = num_in_features
        self.num_out_features = num_out_features
        self.enable_bias = enable_bias

        # num_out_features在前是为了和pytorch的实现保持一致，方便检查计算正确性
        self.weight = nn.Parameter(torch.empty(self.num_out_features, self.num_in_features))
        
        if self.enable_bias:
            self.bias = nn.Parameter(torch.empty(self.num_out_features))

        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.weight, mean=0, std=0.02)
        
        if self.enable_bias:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行线性层的矩阵乘法：将输入与权重相乘，并按需加上偏置。

        Args:
            x: 输入张量，最后一维的大小应为 num_in_features。

        Returns:
            输出张量，最后一维的大小为 num_out_features。
        """
        y = x @ self.weight.T

        if self.enable_bias:
            y = y + self.bias

        return y