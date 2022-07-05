"""
Simple CNN to solve FashionMNIST classification task
"""

import torch
import torch.nn as nn

def _init_weights(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)
  
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(8, 8, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 8, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, (3, 3), (1, 1), (1, 1)),
            nn.ReLU()
        )
        self.classifier = nn.Linear(16 * 7 * 7, 10)

        self.apply(_init_weights)

    def forward(self, x):
        B = x.size(0)
        y = self.backbone(x)
        y = self.classifier(y.view(B, -1))
        return y
