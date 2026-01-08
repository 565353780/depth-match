import torch

from depth_match.Module.depth_matcher import DepthMatcher


def demo():
    source_depth = torch.randn([518, 518, 3])
    target_depth = torch.randn([518, 518, 3])

    matched_depth = DepthMatcher.matchDepth(source_depth, target_depth)

    print(matched_depth.shape)
    return True
