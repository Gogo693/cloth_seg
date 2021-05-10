import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import nn
import time
import torch.nn.functional as F
import numpy as np

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images

class ClothDataset(Dataset):
    '''
    Create a Dataset for face images as triplets: anchor - positive - negative
    '''

    def __init__(self, transform=None):
        #self.cloth_dir = '../ACGPN_landmarks/ACGPN_inference/data/Data_preprocessing/test_color/'
        #self.mask_dir = '../ACGPN_landmarks/ACGPN_inference/data/Data_preprocessing/test_edge/'

        self.cloth_dir = '../ACGPN_landmarks/ACGPN_train/data/train_color/'
        self.mask_dir = '../ACGPN_landmarks/ACGPN_train/data/train_edge/'

        self.transform = transform

        self.cloth_list = []
        self.mask_list = []

        # create triplet lists
        for i, c_name in enumerate(os.listdir(self.cloth_dir)):
            self.cloth_list.append(self.cloth_dir + c_name)
            self.mask_list.append(self.mask_dir + c_name)


    def __len__(self):
        return len(self.cloth_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cloth = io.imread(self.cloth_list[idx])
        mask = io.imread(self.mask_list[idx])

        if self.transform:
            cloth = self.transform(cloth)
            mask = self.transform(mask)

        sample = {
            'cloth': cloth,
            'mask': mask
        }

        return sample

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc*2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()


# Dataset creation
train_dataset = ClothDataset(transform=transforms.Compose([
                                               transforms.ToTensor()
                                           ]))


# Dataloader creation
train_loader = DataLoader(train_dataset, batch_size = 4, shuffle = True, num_workers = 4)

model = UnetGenerator(3, 1, 6, ngf=64, norm_layer=nn.InstanceNorm2d)

device = 'cuda'

model.to(device)


os.makedirs('runs', exist_ok=True)
board = SummaryWriter(log_dir=os.path.join('runs'))

# check parameters to learn
params_to_update = model.parameters()
print("Params to learn:")
params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)

keep_step = 1
decay_step = 1
display_count = 1  #len(train_loader)

# Optimizer creation
# optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -
                                                                                max(0, step - keep_step) / float(
    decay_step + 1))

criterion = nn.BCEWithLogitsLoss()
mean_loss = 0
step = 0

for epoch in range(keep_step + decay_step):
    for n_batch, batch in enumerate(train_loader):
        iter_start_time = time.time()
        inputs = batch

        cloth = inputs['cloth'].to(device)
        mask = inputs['mask'].to(device)

        # input = torch.cat([cloth, mask], 1)
        input = cloth

        outputs = model(input)
        outputs = F.sigmoid(outputs)

        visuals = [cloth, mask, outputs]

        # print(torch.max(outputs))
        # print(torch.max(mask))
        # print(input.size())
        # print(mask.size())
        # print(outputs.size())
        # mask = torch.squeeze(mask.long())
        # mask = mask.long()
        # print(mask.size())
        # loss = F.cross_entropy(outputs, mask)
        loss = criterion(outputs, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #mean_loss += loss.item() / display_count

        if (step + 1) % display_count == 0:
            board.add_scalar('metric', loss.item(), step + 1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f'
                  % (step + 1, t, loss.item()))
            loss = 0

        if (step + 1) % 500 == 0: #(len(train_loader) / 20) == 0:
            #plt.figure()
            #f, ax = plt.subplots(1, 3)
            #plt.tight_layout()
            #ax[0].imshow(cloth[0].cpu().permute(1, 2, 0))
            #ax[1].imshow(torch.cat([mask[0].cpu(), mask[0].cpu(), mask[0].cpu()], 0).permute(1, 2, 0))
            #ax[2].imshow(
            #    torch.cat([outputs[0].detach().cpu(), outputs[0].detach().cpu(), outputs[0].detach().cpu()], 0).permute(
            #        1, 2, 0))

            a = input[0].float().cuda()
            b = torch.cat([mask[0], mask[0], mask[0]], 0).cuda()
            c = torch.cat([outputs[0], outputs[0], outputs[0]], 0).cuda()
            combine = torch.cat([a, b, c], 2).squeeze()
            cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
            board.add_image('combine', (combine.data + 1) / 2.0, step)

            #board_add_images(board, 'combine', visuals, step + 1)

        step += 1


os.makedirs('checkpoints', exist_ok=True)
save_checkpoint(model, os.path.join('./checkpoints/', 'check.pth'))

        # if (step + 1) % opt.save_count == 0:
        #    save_checkpoint(model, os.path.join(opt.checkpoint_dir, 'step_%06d.pth' % (step + 1)))