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
from visualization import save_image
import cv2


def dice_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union

class ClothDataset(Dataset):
    '''
    Create a Dataset for face images as triplets: anchor - positive - negative
    '''

    def __init__(self, transform=None):
        self.cloth_dir = '../ACGPN_landmarks/ACGPN_inference/data/Data_preprocessing/test_color/'
        self.mask_dir = '../ACGPN_landmarks/ACGPN_inference/data/Data_preprocessing/test_edge/'

        #self.cloth_dir = '../ACGPN_landmarks/ACGPN_train/data/test_color/'
        #self.mask_dir = '../ACGPN_landmarks/ACGPN_train/data/test_edge/'

        self.transform = transform

        self.cloth_list = []
        self.mask_list = []
        self.names = []

        # create triplet lists
        for i, c_name in enumerate(os.listdir(self.cloth_dir)):
            self.cloth_list.append(self.cloth_dir + c_name)
            self.mask_list.append(self.mask_dir + c_name)
            self.names.append(c_name)


    def __len__(self):
        return len(self.cloth_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cloth = io.imread(self.cloth_list[idx])
        mask = io.imread(self.mask_list[idx])
        name = self.names[idx]

        if self.transform:
            cloth = self.transform(cloth)
            mask = self.transform(mask)

        sample = {
            'cloth': cloth,
            'mask': mask,
            'name': name
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


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return
    model.load_state_dict(torch.load(checkpoint_path))
    model.cuda()


# Dataset creation
train_dataset = ClothDataset(transform=transforms.Compose([
                                               transforms.ToTensor()
                                           ]))


# Dataloader creation
train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True, num_workers = 1)

model = UnetGenerator(3, 1, 6, ngf=64, norm_layer=nn.InstanceNorm2d)

load_checkpoint(model, './checkpoints/check.pth')

device = 'cuda'

model.to(device)


os.makedirs('runs', exist_ok=True)
board = SummaryWriter(log_dir=os.path.join('runs'))

# check parameters to learn
params_to_update = model.parameters()
#print("Params to learn:")
params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        #print("\t",name)

keep_step = 1
decay_step = 1
display_count = 1  #len(train_loader)

# Optimizer creation
# optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -
  #                                                                              max(0, step - keep_step) / float(
 #   decay_step + 1))

#criterion = nn.BCEWithLogitsLoss()
#mean_loss = 0

step = 0

accuracy = 0

with torch.no_grad():
    for n_batch, batch in enumerate(train_loader):
        iter_start_time = time.time()
        inputs = batch

        cloth = inputs['cloth'].to(device)

        # input = torch.cat([cloth, mask], 1)
        input = cloth

        outputs = model(input)
        outputs = F.sigmoid(outputs)
        outputs[outputs >= 0.4] = 1
        outputs[outputs < 0.4] = 0

        #save_image(outputs, inputs['name'], './results/')

        accuracy += dice_metric(outputs.cpu(), inputs['mask'].cpu()) / len(train_loader)

        combine = torch.cat([outputs, outputs, outputs], 1).squeeze()
        combine = combine.float().cuda()
        cv_img = (combine.permute(1,2,0).detach().cpu().numpy()+1)/2
        rgb = (cv_img * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        n = str(step) + '.png'
        cv2.imwrite('results/' + inputs['name'][0], bgr) #.split('.')[0] + '.png'

        # combine=c[0].squeeze()
        # cv_img=(combine.permute(1,2,0).detach().cpu().numpy()+1)/2
        #cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2

        step += 1


        # if (step + 1) % opt.save_count == 0:
        #    save_checkpoint(model, os.path.join(opt.checkpoint_dir, 'step_%06d.pth' % (step + 1)))

    print(accuracy)