import torch
from Auto_Config.base import AutoCfgFunctional
from Auto_Config.global_config_reader import global_cfg

from OrangeLLM_Core.unit_test import test_correctness_F, test_efficiency_F



def softmax_by_handcode(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    input_shift = input - input.max(dim = dim, keepdim = True).values
    input_exp = torch.exp(input_shift)
    input_exp_sum = torch.sum(input_exp, dim = dim, keepdim = True)
    result = input_exp / input_exp_sum

    return result


def softmax_by_pytorch(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.nn.functional.softmax(input=input, dim=dim)




implement_register = {"basic_by_handcode": softmax_by_handcode,
                      "standard_pytorch": softmax_by_pytorch}

def softmax(input: torch.Tensor, dim: int = -1):
    return AutoCfgFunctional(global_cfg["implement_version"], implement_register, input, dim)

if __name__ == "__main__":
    """
    计算正确性验证 及 计算效率对比
    """
    ########################################################################################
    print("\n计算正确性测试")

    test_in_dim = 8
    test_out_dim = 4
    test_batch_size = 2
    test_dtype = torch.float32
    test_input = torch.randn(test_batch_size, test_in_dim, dtype=test_dtype)

    test_result= test_correctness_F(softmax_by_pytorch, 
                                    [softmax_by_handcode],
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
    test_result= test_efficiency_F(softmax_by_pytorch, 
                                    [softmax_by_handcode],
                                    test_input)
    print("CPU计算:")
    for name, out in test_result.items():
        print(f"{name} 耗时:{out:.8f} \n")
  
    test_result= test_efficiency_F(softmax_by_pytorch, 
                                    [softmax_by_handcode],
                                    test_input,
                                    device="cuda:0")
    print("GPU计算:")
    for name, out in test_result.items():
        print(f"{name} 耗时:{out:.8f} \n")


