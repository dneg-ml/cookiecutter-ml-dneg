"""
Initializing datasets can get complicated. Keep your code clean!
"""

import torchvision
import torchvision.transforms as transforms

def get_dataset(root='./data/FashionMNIST', train=True):
    return torchvision.datasets.FashionMNIST(
        root=root,
        train=train,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()                                 
        ])
    )
