from models.registers import MODULES
import torch
from torch import nn
from models.iscnet.modules.layers import ResnetPointnet


@MODULES.register_module
class ClassEncoder(nn.Module):
    def __init__(self, cfg):
        self.encoder = ResnetPointnet(c_dim=cfg.config['data']['c_dim'],
                                      dim=self.input_feature_dim + 3 + 128,
                                      hidden_dim=cfg.config['data']['hidden_dim'])
        self.linear = torch.Linear(cfg.config['data']['c_dim'], cfg.config['data']['num_classes'])

    def forward(self, pc):
        features = self.encoder(pc)
        logits = self.linear(features)
        return logits, features