import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
import random
import itertools
from tqdm import tqdm
from models import Generator
from models import Discriminator
from utils import AverageMeter, weights_init_normal, ReplayBuffer, ImageDataset, ImageDataset2

bs = 1
image_size = 200
input_nc = 3
output_nc = 3
lr = 1e-4

transforms_ = [
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataset = ImageDataset2('test_data', transforms_=transforms_, unaligned=True)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)

netG_A2B = Generator(input_nc, output_nc)
netG_B2A = Generator(output_nc, input_nc)
netG_A2C = Generator(input_nc, output_nc)
netG_C2A = Generator(output_nc, input_nc)
netG_B2C = Generator(input_nc, output_nc)
netG_C2B = Generator(output_nc, input_nc)
netD_A = Discriminator(input_nc)
netD_B = Discriminator(output_nc)
netD_C = Discriminator(output_nc)

netG_A2B.cuda()
netG_B2A.cuda()
netG_A2C.cuda()
netG_C2A.cuda()
netG_B2C.cuda()
netG_C2B.cuda()
netD_A.cuda()
netD_B.cuda()
netD_C.cuda()

# 加载训练好的模型
netG_A2B.load_state_dict(torch.load('model/white-black-yellow/netG_A2B.pth'))
netG_B2A.load_state_dict(torch.load('model/white-black-yellow/netG_B2A.pth'))
netG_A2C.load_state_dict(torch.load('model/white-black-yellow/netG_A2C.pth'))
netG_C2A.load_state_dict(torch.load('model/white-black-yellow/netG_C2A.pth'))
netG_B2C.load_state_dict(torch.load('model/white-black-yellow/netG_B2C.pth'))
netG_C2B.load_state_dict(torch.load('model/white-black-yellow/netG_C2B.pth'))
netD_A.load_state_dict(torch.load('model/white-black-yellow/netD_A.pth'))
netD_B.load_state_dict(torch.load('model/white-black-yellow/netD_B.pth'))
netD_C.load_state_dict(torch.load('model/white-black-yellow/netD_C.pth'))

Tensor = torch.cuda.FloatTensor
input_A = Tensor(bs, input_nc, image_size, image_size)
input_B = Tensor(bs, output_nc, image_size, image_size)
input_C = Tensor(bs, output_nc, image_size, image_size)
target_real = Variable(Tensor(bs).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(bs).fill_(0.0), requires_grad=False)

for i, batch in enumerate(dataloader):
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))
    real_C = Variable(input_C.copy_(batch['C']))

    # Generate output
    fake_BA = 0.5 * (netG_A2B(real_A).data + 1.0)
    fake_BC = 0.5 * (netG_C2B(real_C).data + 1.0)
    fake_AB = 0.5 * (netG_B2A(real_B).data + 1.0)
    fake_AC = 0.5 * (netG_C2A(real_C).data + 1.0)
    fake_CA = 0.5 * (netG_A2C(real_A).data + 1.0)
    fake_CB = 0.5 * (netG_B2C(real_B).data + 1.0)

    # Save image files
    save_image(0.5 * (real_A + 1), 'output/%04drA.png' % (i + 1))
    save_image(0.5 * (real_B + 1), 'output/%04drB.png' % (i + 1))
    save_image(0.5 * (real_C + 1), 'output/%04drC.png' % (i + 1))
    save_image(fake_AB, 'output/%04dfAB.png' % (i + 1))
    save_image(fake_AC, 'output/%04dfAC.png' % (i + 1))
    save_image(fake_BA, 'output/%04dfBA.png' % (i + 1))
    save_image(fake_BC, 'output/%04dfBC.png' % (i + 1))
    save_image(fake_CA, 'output/%04dfCA.png' % (i + 1))
    save_image(fake_CB, 'output/%04dfCB.png' % (i + 1))
