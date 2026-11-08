# globals.py
import yaml
import os
from pathlib import Path

current_file = Path(__file__).resolve()

global_config_path = current_file.parent.parent / "config.yaml"

def config_load(config_path: str = "config.yaml"):
    """读取全局config配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        global_config = yaml.safe_load(f)
    print("已加载实现配置，当前实现版本: {}".format(global_config["implement_version"]))

    return global_config

# 暴露一个全局单例对象
global_cfg = config_load(global_config_path)