import torch
import torch.nn as nn 
from transformers import RegNetModel


from src.core import register

__all__ = ['RegNet']

@register
class RegNet(nn.Module):
    def __init__(self, configuration, return_idx=[0, 1, 2, 3], pretrained=True):
        super(RegNet, self).__init__()
        if pretrained:
            self.model = RegNetModel.from_pretrained("facebook/regnet-y-040")
            print(f'Load RegNet state_dict from HuggingFace (facebook/regnet-y-040)')
        else:
            self.model = RegNetModel.from_config(RegNetConfig())
            print(f'RegNet initialized randomly (pretrained=False)')
        self.return_idx = return_idx


    def forward(self, x):
        
        outputs = self.model(x, output_hidden_states = True)
        x = outputs.hidden_states[2:5]

        return x