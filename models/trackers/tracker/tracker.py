from torch import nn


class tracker(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg