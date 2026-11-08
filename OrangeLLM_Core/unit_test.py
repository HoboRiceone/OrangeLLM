import torch
import numpy as np
import torch.nn as nn
import time
from typing import List, Type, Dict, Any, Tuple, Optional, Callable

def data_iter(features, labels, batch_size=8):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)  #样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        indexs = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield  features.index_select(0, indexs), labels.index_select(0, indexs)
        

def train_model(model, data_input, data_label, loss_fn, epochs):
    
    print(f"\n{type(model).__name__} train log:")
    for epoch in range(1, epochs + 1):
        loss_list,metric_list = [],[]

        for features, labels in data_iter(data_input, data_label, 20):
            predictions = model.forward(features)
            loss = loss_fn(predictions, labels)
                
            # 反向传播求梯度
            loss.backward()
            
            # 梯度下降法更新参数
            with torch.no_grad():
                for param in model.parameters():
                    param -= 0.01 * param.grad
                
            # 梯度清零
            model.zero_grad()
                
            loss_list.append(loss.item())
        
        loss = np.mean(loss_list)

        print("epoch =",epoch,"loss = ",loss)



def _get_attr_by_path(module: nn.Module, path: str) -> torch.Tensor:
    """根据点分路径字符串递归获取模块中的属性（支持深层嵌套）"""
    attrs = path.split(".")
    obj = module
    for attr in attrs:
        if not hasattr(obj, attr):
            raise AttributeError(
                f"模块 {type(obj).__name__} 中不存在属性 '{attr}'，"
                f"完整路径: '{path}'，已解析到: {type(obj).__name__}"
            )
        obj = getattr(obj, attr)
    return obj


def _sync_weights_by_mapping(
    ref_model: nn.Module,
    test_model: nn.Module,
    weight_mapping: Dict[str, str]) -> None:
    """根据映射字典将基准模型的参数复制到测试模型"""
    with torch.no_grad():
        for ref_path, test_path in weight_mapping.items():
            ref_tensor = _get_attr_by_path(ref_model, ref_path)
            test_tensor = _get_attr_by_path(test_model, test_path)

            if ref_tensor.shape != test_tensor.shape:
                raise ValueError(
                    f"参数形状不匹配！\n"
                    f"  基准 [{ref_path}]: {ref_tensor.shape}\n"
                    f"  测试 [{test_path}]: {test_tensor.shape}"
                )
            test_tensor.copy_(ref_tensor)


def _instantiate_and_sync(
    ref_class: Type[nn.Module],
    test_classes: List[Type[nn.Module]],
    hyperparams: Dict[str, Any],
    seed: int,
    sync_weights: bool = True,
    weight_mapping: Optional[Dict[str, str]] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32) -> Tuple[nn.Module, List[nn.Module]]:
    """
    实例化基准模型和测试模型，并根据配置同步可学习参数。

    Args:
        ref_class:      基准模型类
        test_classes:   测试模型类列表
        hyperparams:    超参数字典，直接 **kwargs 传入构造函数
        sync_weights:   是否进行参数同步
                        - True:  执行同步（默认）
                        - False: 跳过同步，各模型保持独立随机初始化
        weight_mapping: 参数路径映射字典 {ref_path: test_path}
                        - 当 sync_weights=True 时生效
                        - 若为 None: 自动按同名属性同步 weight/bias
                        - 若为 dict: 严格按映射同步
        seed:           随机种子
        device:         设备
        dtype:          数据类型

    Returns:
        (ref_model, [test_model_1, test_model_2, ...])
    """
    torch.manual_seed(seed)

    # 1. 实例化所有模型
    ref_model = ref_class(**hyperparams).to(dtype).to(device)
    test_models = [
        cls(**hyperparams).to(dtype).to(device) for cls in test_classes
    ]

    # 2. 仅在 sync_weights=True 时执行同步
    if sync_weights:
        if weight_mapping is None:
            # 默认行为：自动按 named_parameters 同名匹配
            effective_mapping = {name: name for name, _ in ref_model.named_parameters()}
        else:
            effective_mapping = weight_mapping

        if effective_mapping:
            for model in test_models:
                _sync_weights_by_mapping(ref_model, model, effective_mapping)

    # 3. 设置 eval 模式
    ref_model.eval()
    for model in test_models:
        model.eval()

    return ref_model, test_models


# =============================================
# 测试功能 1：计算正确性验证
# =============================================
def test_correctness(
    ref_class: Type[nn.Module],
    test_classes: List[Type[nn.Module]],
    hyperparams: Dict[str, Any],
    weight_mapping: Dict[str, str],
    test_input: torch.Tensor,
    seed: int = 689,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    计算正确性验证：对比基准模型与各测试模型的前向传播输出。

    参数:
        ref_class:    基准模型类 (如 nn.Linear)
        test_classes: 测试模型类列表 (如 [Linear_by_handcode, Linear_by_pytorch_kernal])
        hyperparams:  超参数字典，可直接解包用于实例化模型 (如 {'in_features': 8, 'out_features': 4, 'bias': True})
        test_input:   测试输入张量
        seed:         随机种子
        device:       设备
        dtype:        数据类型

    返回:
        Dict[str, torch.Tensor]: 包含所有模型输出的字典，key 为模型类名，value 为输出张量
    """
    ref_model, test_models = _instantiate_and_sync(
        ref_class, test_classes, hyperparams, seed, True, weight_mapping, device, dtype
    )
    test_input = test_input.to(device).to(dtype)

    results = {}

    # 基准模型前向传播
    with torch.no_grad():
        ref_out = ref_model(test_input)
    results[ref_class.__name__] = ref_out

    # 测试模型前向传播
    for cls, model in zip(test_classes, test_models):
        with torch.no_grad():
            out = model(test_input)
        results[cls.__name__] = out

    return results


def test_correctness_F(
    ref_func: Callable,
    test_funcs: List[Callable],
    test_input: torch.Tensor,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    计算正确性验证：对比基准纯函数与各测试纯函数的计算结果。
    
    参数:
        ref_func:    基准函数 (如 torch.relu)
        test_funcs:  测试函数列表 (如 [my_relu])
        test_input:  唯一的输入张量
        device:      设备
        dtype:       数据类型

    返回:
        Dict[str, torch.Tensor]: key 为函数名，value 为输出张量
    """
    # 1. 统一输入数据的设备和类型
    test_input = test_input.to(device=device, dtype=dtype)
    
    results = {}

    # 2. 基准函数前向传播
    with torch.no_grad():
        ref_out = ref_func(test_input)
    results[ref_func.__name__] = ref_out

    # 3. 测试函数前向传播
    for func in test_funcs:
        with torch.no_grad():
            out = func(test_input)
        results[func.__name__] = out

    return results

# =============================================
# 测试功能 2：计算效率对比
# =============================================
def test_efficiency(
    ref_class: Type[nn.Module],
    test_classes: List[Type[nn.Module]],
    hyperparams: Dict[str, Any],
    weight_mapping: Dict[str, str],
    test_input: torch.Tensor,
    seed: int = 689,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    warmup_iters: int = 10,
    timing_iters: int = 100,
) -> Dict[str, float]:
    """
    计算效率对比：测量基准模型与各测试模型的前向传播耗时。

    参数:
        ref_class:    基准模型类
        test_classes: 测试模型类列表
        hyperparams:  超参数字典
        test_input:   测试输入张量
        seed:         随机种子
        device:       设备 ("cpu" 或 "cuda:0")
        dtype:        数据类型
        warmup_iters:  预热迭代次数（避免首次运行的开销影响结果）
        timing_iters:  计时迭代次数（多次测量取平均）

    返回:
        Dict[str, float]: 包含各模型平均耗时的字典，key 为模型类名，value 为平均耗时(秒)
    """
    ref_model, test_models = _instantiate_and_sync(
        ref_class, test_classes, hyperparams, seed, True, weight_mapping, device, dtype
    )
    test_input = test_input.to(device).to(dtype)

    all_models = [(ref_class.__name__, ref_model)] + [
        (cls.__name__, model) for cls, model in zip(test_classes, test_models)
    ]

    elapsed_times = {}

    for name, model in all_models:
        # 预热
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = model(test_input)

        # GPU 需要同步
        if "cuda" in device:
            torch.cuda.synchronize()

        # 计时
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(timing_iters):
                _ = model(test_input)

        if "cuda" in device:
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        elapsed_times[name] = elapsed / timing_iters  # 取平均

    return elapsed_times


def test_efficiency_F(
    ref_func: Callable,
    test_funcs: List[Callable],
    test_input: torch.Tensor,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    warmup_iters: int = 10,
    timing_iters: int = 100,
) -> Dict[str, float]:
    """
    计算效率对比：测量基准函数与各测试函数的前向传播耗时。
    
    参数:
        ref_func:    基准函数 (如 torch.relu)
        test_funcs:  测试函数列表 (如 [my_relu])
        test_input:  唯一的输入张量
        device:      设备
        dtype:       数据类型
        warmup_iters:  预热迭代次数（避免首次运行的开销影响结果）
        timing_iters:  计时迭代次数（多次测量取平均）

    返回:
        Dict[str, float]: 包含各函数平均耗时的字典，key 为函数名，value 为平均耗时(秒)
    """
    # 1. 统一输入数据的设备和类型
    test_input = test_input.to(device=device, dtype=dtype)
    
    elapsed_times = {}

    all_funcs = [(ref_func.__name__, ref_func)] + [(func.__name__, func) for func in test_funcs]

    for name, func in all_funcs:
        # 预热
        with torch.no_grad():
            for _ in range(warmup_iters):
                out = func(test_input)

        # GPU 需要同步
        if "cuda" in device:
            torch.cuda.synchronize()

        # 计时
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(timing_iters):
                out = func(test_input)

        if "cuda" in device:
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        elapsed_times[name] = elapsed / timing_iters  # 取平均

    return elapsed_times
    


# =============================================
# 测试功能 3：拟合正确性验证
# =============================================
def test_fitting(
    ref_class: Type[nn.Module],
    test_classes: List[Type[nn.Module]],
    hyperparams: Dict[str, Any],
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    loss_fn: nn.Module,
    epochs: int = 10,
    seed: int = 689,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Dict[str, Any]]:
    """
    拟合正确性验证：训练基准模型与各测试模型，并在验证集上评估拟合效果。

    参数:
        ref_class:    基准模型类
        test_classes: 测试模型类列表
        hyperparams:  超参数字典
        train_X:      训练集输入
        train_y:      训练集标签
        val_X:        验证集输入
        val_y:        验证集标签
        loss_fn:      损失函数
        lr:           学习率
        epochs:       训练轮数
        seed:         随机种子
        device:       设备
        dtype:        数据类型

    返回:
        Dict[str, Dict[str, Any]]: 包含各模型验证结果的字典
            - 'predictions': 验证集预测结果张量
            - 'mse':         验证集 MSE 指标
            - 'val_labels':  验证集真实标签（仅返回一次，所有模型共享）
    """
    # 数据迁移到设备
    train_X = train_X.to(device).to(dtype)
    train_y = train_y.to(device).to(dtype)
    val_X = val_X.to(device).to(dtype)

    # 实例化所有模型（各自独立初始化，不同步权重）
    all_models = []

    torch.manual_seed(seed)
    ref_model = ref_class(**hyperparams).to(dtype).to(device)
    all_models.append((ref_class.__name__, ref_model))

    for cls in test_classes:
        torch.manual_seed(seed)  # 每个测试类也用相同seed独立初始化，保证各自可复现但不互相复制权重
        model = cls(**hyperparams).to(dtype).to(device)
        all_models.append((cls.__name__, model))

    # 数据迁移到设备
    train_X = train_X.to(device).to(dtype)
    train_y = train_y.to(device).to(dtype)
    val_X = val_X.to(device).to(dtype)

    # 训练所有模型
    for name, model in all_models:
        train_model(model, train_X, train_y, loss_fn, epochs)
        model.eval()

    # 验证集前向传播
    results = {}
    for name, model in all_models:
        with torch.no_grad():
            preds = model(val_X)
        results[name] = {
            "predictions": preds
        }

    return results
