import torch
import torch.nn as nn
import torch.nn.functional as F
from Auto_Config.base import AutoCfgModule
from Auto_Config.global_config_reader import global_cfg
from OrangeLLM_Core.unit_test import test_correctness, test_efficiency, test_fitting


class Linear_by_handcode(nn.Module):

    def __init__(self, num_in_features: int, num_out_features: int, enable_bias: bool) -> None:
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


class Linear_by_pytorch_kernal(nn.Module):

    def __init__(self, num_in_features: int, num_out_features: int, enable_bias: bool) -> None:
        """初始化线性层，创建权重参数，并按需创建偏置参数。

        Args:
            num_in_features: 输入特征数量。
            num_out_features: 输出特征数量。
            enable_bias: 是否启用偏置。
        """
        super(Linear_by_pytorch_kernal, self).__init__()
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
        """调用pytorch functional完成计算。

        Args:
            x: 输入张量，最后一维的大小应为 num_in_features。

        Returns:
            输出张量，最后一维的大小为 num_out_features。
        """
        bias = self.bias if self.enable_bias else None
        y = F.linear(input=x, weight=self.weight, bias=bias)

        return y

class Linear_pytorch(nn.Linear):
    def __init__(self, num_in_features: int, num_out_features: int, enable_bias: bool) -> None:
        super().__init__(in_features=num_in_features,out_features=num_out_features,bias=enable_bias)


class Linear(AutoCfgModule):
    # 显式绑定：该模块的版本由全局配置控制，或者在特别指定
    current_implement_id = global_cfg["implement_version"]
    
    implement_register = {
        "basic_by_handcode": Linear_by_handcode,
        "speedup_by_pytorch_kernal": Linear_by_pytorch_kernal,
        "standard_pytorch": Linear_pytorch
    }


if __name__ == "__main__":
    """
    计算正确性验证 及 计算效率对比
    """
    weight_mapping = {"weight": "weight", "bias": "bias"}
    ########################################################################################
    print("\n计算正确性测试")

    test_in_dim = 8
    test_out_dim = 4
    test_batch_size = 2
    test_dtype = torch.float32
    test_input = torch.randn(test_batch_size, test_in_dim, dtype=test_dtype)
    test_seed = 44

    model_hyperparams = {"num_in_features": test_in_dim, "num_out_features": test_out_dim, "enable_bias": True}
    weight_mapping = {"weight": "weight", "bias": "bias"}
    test_result= test_correctness(Linear_pytorch, 
                     [Linear_by_handcode, Linear_by_pytorch_kernal],
                     model_hyperparams,
                     weight_mapping,
                     test_input)

    for name, out in test_result.items():
        print(f"{name} 输出形状: {out.shape}, 数值:\n {out.data}\n")

    ########################################################################################
    print("\n计算效率测试")

    test_in_dim = 1024
    test_out_dim = 1024
    test_batch_size = 8
    test_input = torch.randn(test_batch_size, test_in_dim, dtype=test_dtype)
    
    model_hyperparams = {"num_in_features": test_in_dim, "num_out_features": test_out_dim, "enable_bias": True}
    test_result = test_efficiency(Linear_pytorch, 
                                [Linear_by_handcode, Linear_by_pytorch_kernal],
                                model_hyperparams,
                                weight_mapping,
                                test_input,
                                device="cpu")
    print("CPU计算:")
    for name, out in test_result.items():
        print(f"{name} 耗时:{out:.8f} \n")

    test_result = test_efficiency(Linear_pytorch, 
                            [Linear_by_handcode, Linear_by_pytorch_kernal],
                            model_hyperparams,
                            weight_mapping,
                            test_input,
                            device="cuda:0")
    print("GPU计算:")
    for name, out in test_result.items():
        print(f"{name} 耗时:{out:.8f} \n")
    
    ########################################################################################
    print("\n拟合正确性测试")

    test_in_dim = 8
    test_out_dim = 1
    test_batch_size = 500
    NOISE_STD = 0.5
    test_train_epoch = 20

    W_true = torch.randn(test_in_dim, 1) * 2.0  
    b_true = torch.tensor([1.5])                

    test_X = torch.randn(test_batch_size, test_in_dim)               
    noise = torch.randn(test_batch_size, test_out_dim) * NOISE_STD
    test_Y = test_X @ W_true + b_true + noise                            

    split = int(test_batch_size * 0.8)
    indices = torch.randperm(test_batch_size)
    train_idx, test_idx = indices[:split], indices[split:]

    test_X_train, test_y_train = test_X[train_idx], test_Y[train_idx]
    test_X_val, test_y_val = test_X[test_idx], test_Y[test_idx]

    model_hyperparams = {"num_in_features": test_in_dim, "num_out_features": test_out_dim, "enable_bias": True}
    test_result = test_fitting(Linear_pytorch, 
                            [Linear_by_handcode, Linear_by_pytorch_kernal],
                            model_hyperparams,
                            test_X_train, test_y_train, test_X_val, 
                            nn.MSELoss())

    print("\n验证集真实label:\n", test_y_val.T)
    for name, out in test_result.items():
            print("{} 预测结果:\n {}\n评价指标:{}\n".format(name, out["predictions"].T, nn.MSELoss()(test_y_val, out["predictions"]).item()))   
    


