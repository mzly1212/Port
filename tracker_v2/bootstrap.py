# -*- coding: utf-8 -*-
"""
bootstrap.py — 将项目根目录加入 sys.path, 使 v2 模块可以只读复用
根目录的 I/O 契约 (data.py / config.py / input_adapter / output_adapter / map/)。
不在根目录上做任何写入。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
