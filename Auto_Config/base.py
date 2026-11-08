# modules/base.py
import torch.nn as nn

class AutoCfgModule(nn.Module):
    """多版本自动配置统一基类"""
    
    implement_register = {}
    current_implement_id = None 

    def __new__(cls, *args, **kwargs):
        if cls.current_implement_id:
            
            current_implement_cls = cls.implement_register.get(cls.current_implement_id)
            
            return current_implement_cls(*args, **kwargs)
        else:
            raise TypeError(" 定义模块时未引入当前所配置的名称")


def AutoCfgFunctional(cfg_key, registry, *args, **kwargs):
    """函数版多实现路由"""
    func = registry.get(cfg_key)

    if func is None:
        raise ValueError(
            f"配置 '{cfg_key}' 无效\n"
        )
    return func(*args, **kwargs)