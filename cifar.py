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

def compute_variance(tricked=False):
    if tricked:
        return 4.4245717472233694e-06
    train_data, _ = get_loader(1, False, False, False)
    data = np.zeros((len(cifar100), 32, 32, 3))
    for id, img in enumerate(train_data):
        data[id] = img
    return np.var(data / 255.0)


def get_loader(batch, var=False, shuffle=True, drop_last=True):
    return torch.utils.data.DataLoader(cifar100, batch_size=batch, shuffle=shuffle,
                                         collate_fn=collate, drop_last=drop_last), \
           compute_variance(tricked=True) if var else None





