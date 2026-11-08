import torch.nn as nn
from Auto_Config.base import AutoCfgModule
from Auto_Config.global_config_reader import global_cfg

class _ToyOp_A(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1

class _ToyOp_B(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 100

class ToyOp(AutoCfgModule):
    # 显式绑定：该模块的版本由全局配置控制，或者在特别指定
    current_implement_id = global_cfg["implement_version"]
    
    implement_register = {
        "basic_by_handcode": _ToyOp_A,
        "basic_by_handcode1": _ToyOp_B,
    }