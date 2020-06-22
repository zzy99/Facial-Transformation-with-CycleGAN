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
epochs = 100

transforms_ = [
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataset = ImageDataset2('data', transforms_=transforms_, unaligned=True)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)  # num_workers=4)

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

# netG_A2B.apply(weights_init_normal)
# netG_B2A.apply(weights_init_normal)
# netG_A2C.apply(weights_init_normal)
# netG_C2A.apply(weights_init_normal)
# netG_B2C.apply(weights_init_normal)
# netG_C2B.apply(weights_init_normal)
# netD_A.apply(weights_init_normal)
# netD_B.apply(weights_init_normal)
# netD_C.apply(weights_init_normal)

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

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=lr)
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr)
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr)

Tensor = torch.cuda.FloatTensor
input_A = Tensor(bs, input_nc, image_size, image_size)
input_B = Tensor(bs, output_nc, image_size, image_size)
input_C = Tensor(bs, output_nc, image_size, image_size)
target_real = Variable(Tensor(bs).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(bs).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

loss1 = AverageMeter()
loss2 = AverageMeter()
loss3 = AverageMeter()
loss4 = AverageMeter()
loss5 = AverageMeter()

# #two domain
# for epoch in range(epochs):
#     tk = tqdm(enumerate(dataloader), total=len(dataloader), position=0, leave=True)
#
#     for i, batch in tk:
#         real_A = Variable(input_A.copy_(batch['A']))
#         real_B = Variable(input_B.copy_(batch['B']))
#
#         ###### Generators A2B and B2A ######
#         optimizer_G.zero_grad()
#
#         # Identity loss
#         same_B = netG_A2B(real_B)
#         loss_identity_B = criterion_identity(same_B, real_B) * 5.0
#
#         same_A = netG_B2A(real_A)
#         loss_identity_A = criterion_identity(same_A, real_A) * 5.0
#
#         # GAN loss
#         fake_B = netG_A2B(real_A)
#         pred_fake = netD_B(fake_B)
#         loss_GAN_A2B = criterion_GAN(pred_fake, target_real)
#
#         fake_A = netG_B2A(real_B)
#         pred_fake = netD_A(fake_A)
#         loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
#
#         # Cycle loss
#         recovered_A = netG_B2A(fake_B)
#         loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0
#
#         recovered_B = netG_A2B(fake_A)
#         loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0
#
#         # Total loss
#         loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
#         loss_G.backward()
#
#         optimizer_G.step()
#         ###################################
#
#         ###### Discriminator A ######
#         optimizer_D_A.zero_grad()
#
#         # Real loss
#         pred_real = netD_A(real_A)
#         loss_D_real = criterion_GAN(pred_real, target_real)
#
#         # Fake loss
#         fake_A = fake_A_buffer.push_and_pop(fake_A)
#         pred_fake = netD_A(fake_A.detach())
#         loss_D_fake = criterion_GAN(pred_fake, target_fake)
#
#         # Total loss
#         loss_D_A = (loss_D_real + loss_D_fake) * 0.5
#         loss_D_A.backward()
#
#         optimizer_D_A.step()
#         ###################################
#
#         ###### Discriminator B ######
#         optimizer_D_B.zero_grad()
#
#         # Real loss
#         pred_real = netD_B(real_B)
#         loss_D_real = criterion_GAN(pred_real, target_real)
#
#         # Fake loss
#         fake_B = fake_B_buffer.push_and_pop(fake_B)
#         pred_fake = netD_B(fake_B.detach())
#         loss_D_fake = criterion_GAN(pred_fake, target_fake)
#
#         # Total loss
#         loss_D_B = (loss_D_real + loss_D_fake) * 0.5
#         loss_D_B.backward()
#
#         optimizer_D_B.step()
#         ###################################
#
#         loss1.update(loss_G.item(), real_A.size(0))
#         loss2.update((loss_identity_A + loss_identity_B).item(), real_A.size(0))
#         loss3.update((loss_GAN_A2B + loss_GAN_B2A).item(), real_A.size(0))
#         loss4.update((loss_cycle_ABA + loss_cycle_BAB).item(), real_A.size(0))
#         loss5.update((loss_D_A + loss_D_B).item(), real_A.size(0))
#
#         tk.set_postfix(loss_G=loss1.avg, loss_G_identity=loss2.avg, loss_G_GAN=loss3.avg, loss_G_cycle=loss4.avg,
#                        loss_D=loss5.avg)
#
#     with open('log.txt', 'a')as f:
#         f.write(str(loss1.avg) + ' ' + str(loss5.avg) + '\n')
#
#     # lr_scheduler_G.step()
#     # lr_scheduler_D_A.step()
#     # lr_scheduler_D_B.step()
#
#     torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
#     torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
#     torch.save(netD_A.state_dict(), 'output/netD_A.pth')
#     torch.save(netD_B.state_dict(), 'output/netD_B.pth')

# three domain
for epoch in range(epochs):
    tk = tqdm(enumerate(dataloader), total=len(dataloader), position=0, leave=True)

    for i, batch in tk:
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        real_C = Variable(input_C.copy_(batch['C']))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        same_B1 = netG_A2B(real_B)
        same_B2 = netG_C2B(real_B)
        loss_identity_B = criterion_identity(same_B1, real_B) * 5.0 + criterion_identity(same_B2, real_B) * 5.0

        same_A1 = netG_B2A(real_A)
        same_A2 = netG_C2A(real_A)
        loss_identity_A = criterion_identity(same_A1, real_A) * 2.5 + criterion_identity(same_A2, real_A) * 2.5

        same_C1 = netG_A2C(real_C)
        same_C2 = netG_B2C(real_C)
        loss_identity_C = criterion_identity(same_C2, real_C) * 2.5 + criterion_identity(same_C2, real_C) * 2.5

        # GAN loss
        fake_BA = netG_A2B(real_A)
        pred_fake = netD_B(fake_BA)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_AB = netG_B2A(real_B)
        pred_fake1 = netD_A(fake_AB)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        fake_CA = netG_A2C(real_A)
        pred_fake = netD_C(fake_CA)
        loss_GAN_A2C = criterion_GAN(pred_fake, target_real)

        fake_AC = netG_C2A(real_C)
        pred_fake = netD_A(fake_AC)
        loss_GAN_C2A = criterion_GAN(pred_fake, target_real)

        fake_CB = netG_B2C(real_B)
        pred_fake = netD_C(fake_CB)
        loss_GAN_B2C = criterion_GAN(pred_fake, target_real)

        fake_BC = netG_C2B(real_C)
        pred_fake = netD_B(fake_BC)
        loss_GAN_C2B = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A1 = netG_B2A(fake_BA)
        recovered_A2 = netG_C2A(fake_CA)
        loss_cycle_A = criterion_cycle(recovered_A1, real_A) * 5.0 + criterion_cycle(recovered_A2, real_A) * 5.0

        recovered_B1 = netG_A2B(fake_AB)
        recovered_B2 = netG_C2B(fake_CB)
        loss_cycle_B = criterion_cycle(recovered_B1, real_B) * 5.0 + criterion_cycle(recovered_B2, real_B) * 5.0

        recovered_C1 = netG_A2C(fake_AC)
        recovered_C2 = netG_B2C(fake_BC)
        loss_cycle_C = criterion_cycle(recovered_C1, real_C) * 5.0 + criterion_cycle(recovered_C2, real_C) * 5.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_identity_C + \
                 loss_GAN_A2B + loss_GAN_B2A + loss_GAN_A2C + loss_GAN_C2A + loss_GAN_B2C + loss_GAN_C2B + \
                 loss_cycle_A + loss_cycle_B + loss_cycle_C
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_AB = fake_AB_buffer.push_and_pop(fake_AB)
        pred_fake1 = netD_A(fake_AB.detach())
        fake_AC = fake_AC_buffer.push_and_pop(fake_AC)
        pred_fake2 = netD_A(fake_AC.detach())
        loss_D_fake = (criterion_GAN(pred_fake1, target_fake) + criterion_GAN(pred_fake2, target_fake)) * 0.5

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_BA = fake_BA_buffer.push_and_pop(fake_BA)
        pred_fake1 = netD_B(fake_BA.detach())
        fake_BC = fake_BC_buffer.push_and_pop(fake_BC)
        pred_fake2 = netD_B(fake_BC.detach())
        loss_D_fake = (criterion_GAN(pred_fake1, target_fake) + criterion_GAN(pred_fake2, target_fake)) * 0.5

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        ###### Discriminator C ######
        optimizer_D_C.zero_grad()

        # Real loss
        pred_real = netD_C(real_C)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_CA = fake_CA_buffer.push_and_pop(fake_CA)
        pred_fake1 = netD_C(fake_CA.detach())
        fake_CB = fake_CB_buffer.push_and_pop(fake_CB)
        pred_fake2 = netD_C(fake_CB.detach())
        loss_D_fake = (criterion_GAN(pred_fake1, target_fake) + criterion_GAN(pred_fake2, target_fake)) * 0.5

        # Total loss
        loss_D_C = (loss_D_real + loss_D_fake) * 0.5
        loss_D_C.backward()

        optimizer_D_C.step()
        ###################################

        loss1.update(loss_G.item(), real_A.size(0))
        loss2.update((loss_identity_A + loss_identity_B + loss_identity_C).item(), real_A.size(0))
        loss3.update((loss_GAN_A2B + loss_GAN_B2A + loss_GAN_A2C + loss_GAN_C2A + loss_GAN_B2C + loss_GAN_C2B).item(),
                     real_A.size(0))
        loss4.update((loss_cycle_A + loss_cycle_B + loss_cycle_C).item(), real_A.size(0))
        loss5.update((loss_D_A + loss_D_B + loss_D_C).item(), real_A.size(0))

        tk.set_postfix(loss_G=loss1.avg, loss_G_identity=loss2.avg, loss_G_GAN=loss3.avg, loss_G_cycle=loss4.avg,
                       loss_D=loss5.avg)

    with open('log.txt', 'a')as f:
        f.write(str(loss1.avg) + ' ' + str(loss5.avg) + '\n')
    # torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
    # torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
    # torch.save(netD_A.state_dict(), 'output/netD_A.pth')
    # torch.save(netD_B.state_dict(), 'output/netD_B.pth')
