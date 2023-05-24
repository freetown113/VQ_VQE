import torchvision
from torchvision import transforms as tsf
import torch
import numpy as np

img_size = 32

def collate(input):
   image = np.zeros((len(input), img_size, img_size, 3))
   for i, (im, lb) in enumerate(input):
        image[i] = im
   return image

transform = tsf.Compose(
                        [
                         tsf.ToTensor(), 
                         tsf.Resize(img_size, antialias=True),
                         lambda x: x.numpy(), 
                         lambda x: x.transpose(1,2,0),
                         lambda x: x * 2 - 1.0 # put image data in range [-1, 1] for more stable convergence
                        ]
                        )

cifar100 = torchvision.datasets.CIFAR100('./cifar100', train=True, 
                                         transform=transform, 
                                         download=True)

def compute_variance():
    train_data = iter(cifar100)
    data = np.zeros((len(cifar100, 32, 32, 3)))
    for id, (img, lbl) in train_data:
        data[id] = img
    return np.var(data / 255.0)


def get_loader(batch, var=False):
    return torch.utils.data.DataLoader(cifar100, batch_size=batch, shuffle=True,
                                         collate_fn=collate, drop_last=True),
           compute_variance if var else None





