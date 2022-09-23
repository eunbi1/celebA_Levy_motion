
import os
import torch
import tqdm
from tqdm.asyncio import trange, tqdm
from sampler import pc_sampler2, ode_sampler
from torchlevy import LevyStable

levy = LevyStable()
import torchvision.utils as tvu
from Diffusion import VPSDE
from cifar10_model import Model
import glob
import cv2
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, CIFAR10, CelebA, CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from model import *


def testimg(path, png_name='test_tile'):
    path = path + "/*"
    # path = "./levy-only/*"
    file_list = glob.glob(path)

    images = []

    for file in file_list:
        img = cv2.imread(file)
        images.append(img)

    tile_size = 8

    # 바둑판 타일형 만들기
    def img_tile(img_list):
        return cv2.vconcat([cv2.hconcat(img) for img in img_list])

    imgs = []

    for i in range(0, len(images), tile_size):
        assert tile_size**2 == len(images), "invalid tile size"
        imgs.append(images[i:i+tile_size])

    img_tile_res = img_tile(imgs)
    png_name = "./" + png_name + ".png"
    cv2.imwrite(png_name, img_tile_res)


def sample_fid(path='/home/eunbiyoon/comb_Levy_motion/CelebAbatch128lr0.0001ch128ch_mult[1, 2, 2, 2, 4]num_res2dropout0.1clamp101.8_0.1_20/ckpt/CelebAbatch128ch128ch_mult[1, 2, 2, 2, 4]num_res2dropout0.1clamp10_epoch85_1.8_0.1_20.pth',
               image_folder='/home/eunbiyoon/comb_Levy_motion/95_18_celebA',
               alpha=1.8, beta_min=0.1, beta_max=20,total_n_samples= 50000,
               num_steps=1000,
               channels=3,
               image_size=64, batch_size=64, device='cuda', datasets= 'CelebA',
               ch=128, ch_mult=[1, 2, 2, 2,4],
               sampler = "pc_sampler2",
               num_res_blocks=2,
               ):
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)

    total_n_samples = total_n_samples # total num of datasamples (cifar10 has 50000 training dataset)
    n_rounds = total_n_samples  // batch_size

    sde = VPSDE(alpha=alpha, beta_min=beta_min, beta_max=beta_max)

    if datasets == "CIFAR10":
        score_model = Model(ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks)

    if datasets == "CIFAR100":
        score_model = Model(ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks)

    if datasets == "CelebA":
        score_model = Model(ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks)

    if datasets == "MNIST":
        score_model = Unet(
            dim=28,
            channels=1,
            dim_mults=(1, 2, 4,))

    score_model = score_model.to(device)
    if path:
        ckpt = torch.load(path, map_location=device)
        score_model.load_state_dict(ckpt, strict=False)
        score_model.eval()

    j=0
    with torch.no_grad():
        for _ in trange(n_rounds, desc="Generating image samples for FID evaluation."):
            n = batch_size

            if sampler =="ode_sampler":
                samples = ode_sampler(score_model,
                sde,
                sde.alpha,
                batch_size,
                num_steps=num_steps,

                device='cuda',

                clamp = 10,
                initial_clamp =10, final_clamp = 10,
                datasets=datasets, clamp_mode = 'constant')
            if sampler =="pc_sampler2":
                samples = pc_sampler2(score_model,
                                sde,
                                sde.alpha,
                                batch_size,
                                num_steps=num_steps,

                                device='cuda', trajectory=True,

                                clamp=10,
                                initial_clamp=10, final_clamp=10,
                                datasets=datasets, clamp_mode='constant')
            for i in range(len(samples)):
                if i %25 == 0:
                    x = samples[i]
                    x = (x+1)/2
                    x = x.clamp(0.0, 1.0)

                    for j in range(n):
                     sam = x[i]
                     plt.figure(figsize=(1, 1))
                     plt.axis('off')
                     plt.imshow(sam.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
                     dir_path =os.path.join(image_folder, str(i))
                     if not os.path.isdir(dir_path):
                         os.mkdir(dir_path)
                     name = str(f'{j}') + '.png'
                     name = os.path.join(dir_path, name)

                     plt.savefig(name, dpi=500)
                     #plt.savefig(name, dpi=500)
                     plt.cla()
                     plt.clf()




def cifar102png(path='/scratch/private/eunbiyoon/Levy_motion/cifar10'):
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                      transforms.ToTensor()
                                      ])
    dataset = CIFAR10('/scratch/private/eunbiyoon/data', train=True, transform=transform, download=True)
    data_loader = DataLoader(dataset, batch_size=64,
                             shuffle=True, generator=torch.Generator(device='cuda'))
    j=0
    for x,y in tqdm.tqdm(data_loader):
        x = x.to('cuda')
        n = len(x)
        for i in range(n):
            sam = x[i]
            plt.figure(figsize=(1, 1))
            plt.axis('off')
            plt.imshow(sam.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
            name = str(f'{j}') + '.png'
            name = os.path.join(path, name)
            plt.savefig(name)
            # plt.savefig(name, dpi=500)
            plt.cla()
            plt.clf()
            j = j + 1

def mnist2png(path='/scratch/private/eunbiyoon/Levy_motion/mnist'):
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                      transforms.ToTensor()
                                      ])
    dataset = MNIST('/scratch/private/eunbiyoon/data', train=True, transform=transform, download=True)
    data_loader = DataLoader(dataset, batch_size=64,
                             shuffle=True, generator=torch.Generator(device='cuda'))
    j=0
    for x,y in tqdm.tqdm(data_loader):
        x = x.to('cuda')
        n = len(x)
        for i in range(n):
            sam = x[i]
            plt.figure(figsize=(1, 1))
            plt.axis('off')
            plt.imshow(sam.permute(1, 2, 0).cpu(), vmin=0., vmax=1., cmap='gray')
            name = str(f'{j}') + '.png'
            name = os.path.join(path, name)
            plt.savefig(name)
            # plt.savefig(name, dpi=500)
            plt.cla()
            plt.clf()
            j = j + 1
sample_fid()