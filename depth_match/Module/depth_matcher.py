import torch
import numpy as np
from typing import Union

from depth_match.Method.data import toTensor

class DepthMatcher(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def matchDepth(
        source_depth: Union[torch.Tensor, np.ndarray, list],
        target_depth: Union[torch.Tensor, np.ndarray, list],
        target_depth_conf: Union[torch.Tensor, np.ndarray, list],
        dtype=torch.float64,
        device: str='cpu',
    ) -> torch.Tensor:
        """通过加权线性最小二乘实现source_depth到target_depth的配准。

        找到 scale 和 shift 使得 aligned_depth = scale * source_depth + shift
        最小化加权误差: sum(conf * (aligned_depth - target_depth)^2)

        Args:
            source_depth: 源深度图
            target_depth: 目标深度图
            target_depth_conf: 目标深度的置信度，越大表示越可信
            dtype: 数据类型
            device: 计算设备

        Returns:
            配准后的深度图
        """
        source_depth = toTensor(source_depth, dtype, device)
        target_depth = toTensor(target_depth, dtype, device)
        target_depth_conf = toTensor(target_depth_conf, dtype, device)

        # 展平
        s = source_depth.flatten()
        t = target_depth.flatten()
        conf = target_depth_conf.flatten()

        # 创建有效点掩码：过滤掉 <=0 或 >=1e5 的点
        valid_mask = (s > 0) & (s < 1e5) & (t > 0) & (t < 1e5)

        # 获取有效点
        s_valid = s[valid_mask]
        t_valid = t[valid_mask]
        conf_valid = conf[valid_mask]

        # 归一化conf到0-1
        conf_min = conf_valid.min()
        conf_max = conf_valid.max()
        if conf_max > conf_min:
            w = (conf_valid - conf_min) / (conf_max - conf_min)
        else:
            w = torch.ones_like(conf_valid)

        # 加权最小二乘求解 scale 和 shift
        # 目标: min sum(w * (scale * s + shift - t)^2)
        # 正规方程:
        # [sum(w*s^2), sum(w*s) ] [scale]   [sum(w*s*t)]
        # [sum(w*s),   sum(w)   ] [shift] = [sum(w*t)  ]
        ws2 = (w * s_valid * s_valid).sum()
        ws = (w * s_valid).sum()
        wsum = w.sum()
        wst = (w * s_valid * t_valid).sum()
        wt = (w * t_valid).sum()

        # 解2x2线性方程组 (Cramer法则)
        det = ws2 * wsum - ws * ws
        if det.abs() < 1e-10:
            # 退化情况，返回原始深度
            scale = torch.tensor(1.0, dtype=dtype, device=device)
            shift = torch.tensor(0.0, dtype=dtype, device=device)
        else:
            scale = (wsum * wst - ws * wt) / det
            shift = (ws2 * wt - ws * wst) / det

        # 应用配准变换
        aligned_depth = scale * source_depth + shift

        # 过滤掉source_depth中不合适的点，置为0
        invalid_mask = (source_depth <= 0) | (source_depth >= 1e5)
        aligned_depth[invalid_mask] = 0

        return aligned_depth
