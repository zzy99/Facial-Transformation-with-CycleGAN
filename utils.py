import os
import glob
from PIL import Image
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))

    def __getitem__(self, index):
        if len(self.files_A) > len(self.files_B):
            item_A = self.transform(Image.open(self.files_A[random.randint(0, len(self.files_A) - 1)]))
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
        else:
            item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))


class ImageDataset2(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))
        self.files_C = sorted(glob.glob(os.path.join(root, '%sC' % mode) + '/*.*'))

    def __getitem__(self, index):
        if len(self.files_A) > len(self.files_B):
            item_A = self.transform(Image.open(self.files_A[random.randint(0, len(self.files_A) - 1)]))
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
            item_C = self.transform(Image.open(self.files_C[random.randint(0, len(self.files_C) - 1)]))
        else:
            item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
            item_C = self.transform(Image.open(self.files_C[random.randint(0, len(self.files_C) - 1)]))

        return {'A': item_A, 'B': item_B, 'C': item_C}

    def __len__(self):
        return min(len(self.files_A), len(self.files_B), len(self.files_C))
