import copy
from typing import Dict

import cv2
import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms

from gym.spaces import Box
import os
import pathlib

import math
import noise

import matplotlib.pyplot as plt

from dynaphos.cortex_models import \
    get_visual_field_coordinates_from_cortex_full
from dynaphos.simulator import GaussianSimulator
from dynaphos.utils import (load_params, to_numpy, load_coordinates_from_yaml,
                            Map)
# Dynaphos is a library developed in Neural Coding lab, check paper and git repository.

from habitat import get_config
from habitat.core import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.utils.common import get_image_height_width

savepath= '/home/carsan/Data/habitatai/images/bin/'


class ConvLayer2(nn.Module):
    def __init__(self, n_input, n_output,  k_size=3, stride=1, padding=1):
        super(ConvLayer2, self).__init__()

        self.conv = nn.Conv2d(in_channels=n_input, out_channels=n_output,  kernel_size=k_size, stride=stride, padding=padding, bias=False)
        self.swish = nn.SiLU() #nn.Swish()

        # Weight initialization to prevent vanishing gradients
        init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        out = self.swish(self.conv(x))
        return out


class ResidualBlock2(nn.Module):
    def __init__(self, n_channels, stride=1, resample_out=None):
        super(ResidualBlock2, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.swish = nn.SiLU() #nn.Swish()
        self.conv2 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)

        # Weight initialization to prevent vanishing gradients
        init.xavier_uniform_(self.conv1.weight)
        init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        residual = x
        out = self.swish(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = self.swish(out)
        return out


class E2E_Encoder(nn.Module):
    """
    Simple non-generic encoder class that receives 128x128 input and outputs 32x32 feature map as stimulation protocol
    """
    def __init__(self, in_channels=1, out_channels=1,binary_stimulation=True):
        super(E2E_Encoder, self).__init__()

        self.binary_stimulation = binary_stimulation

        self.convlayer1 = ConvLayer2(in_channels,8,3,1,1)
        self.convlayer2 = ConvLayer2(8,16,3,1,1)
        self.maxpool1 = nn.MaxPool2d(2) # Not trained
        self.convlayer3 = ConvLayer2(16,32,3,1,1)
        self.maxpool2 =nn.MaxPool2d(2) # Not trained
        self.res1 = ResidualBlock2(32) # 4 trainable parameters each
        self.res2 = ResidualBlock2(32) # 4 trainable parameters each
        self.res3 = ResidualBlock2(32) # 4 trainable parameters each
        self.res4 = ResidualBlock2(32) # 4 trainable parameters each
        self.convlayer4 =ConvLayer2(32,16,3,1,1)
        self.encconv1 = nn.Conv2d(16,out_channels,3,1,1) #bias true
        self.tanh1 = nn.Tanh()

    def forward(self, x):
        # print('encoder input range', x.min(),x.max(), x.shape)

        # Save input Image
        # Conditional to only save images when first dimension is the number of env and not 512?
        # if x.shape[2] == 128:
            # plt.imsave(savepath + 'enc_input.png', x[0, 0, :, :].detach().cpu().numpy(), cmap=plt.cm.gray)

        x = self.convlayer1(x)
        x = self.maxpool1(self.convlayer2(x))
        x = self.maxpool2(self.convlayer3(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.convlayer4(x)
        x = self.tanh1(self.encconv1(x))

        stimulation = .5*(x+1)
        # print('enc_output',stimulation.shape)
        # plt.imsave(savepath+'enc_output.png', stimulation[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray)
        return stimulation

@baseline_registry.register_obs_transformer()
class Encoder(ObservationTransformer):
    def __init__(self, in_channels=1, out_channels=1,binary_stimulation=True):
        super().__init__()

        self.binary_stimulation = binary_stimulation

        self.transformed_sensor = 'rgb'

        self.model = E2E_Encoder()

        checkpoint_dir = ("/scratch/big/home/carsan/Data/phosphenes/habitat/checkpoints/"
                          "train_NPT_RGB_E2E_2.8Msteps_6Envs_800spe_xavierWeights_lr=e-4_32spu_weightLossPPO=0.1"
                          "/latest.pth")
        state_dict_encoder = torch.load(checkpoint_dir)["state_dict_encoder"]
        self.model.load_state_dict(
            {  # type: ignore
                k[len("model."):]: v
                for k, v in state_dict_encoder.items()
                if "model" in k
            }
        )
        self.model.to("cuda:0")


    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        observations = self._transform_obs(observations)
        return observations

    def _transform_obs(self, observation: torch.Tensor):
        # print('Input of encoder observation shape', observation.shape)

        stimulation = self.model.forward(observation.permute(0,3,1,2).float().cuda())

        # stimulation = .5*(frame+1)  # Is this important?
        stimulation = stimulation.permute(0,2,3,1)

        return stimulation # Continuous space, not only 0s and 1s

    @classmethod
    def from_config(cls, config: get_config):
        return cls()


class Decoder(nn.Module):
    """
    Simple non-generic phosphene decoder.
    in: (256x256) SPV representation
    out: (128x128) Reconstruction
    """
    def __init__(self, in_channels=1, out_channels=1, out_activation='sigmoid'):
        super(Decoder, self).__init__()

        # Activation of output layer
        self.out_activation = {'tanh': nn.Tanh(),
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.LeakyReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]

        self.convlayer1=ConvLayer2(in_channels,16,3,1,1)
        self.convlayer2=ConvLayer2(16,32,3,1,1)
        self.convlayer3=ConvLayer2(32,64,3,2,1)
        self.res1=ResidualBlock2(64) # 4 trainable parameters
        self.res2=ResidualBlock2(64) # 4 trainable parameters
        self.res3=ResidualBlock2(64) # 4 trainable parameters
        self.res4=ResidualBlock2(64) # 4 trainable parameters
        self.convlayer4=ConvLayer2(64,32,3,1,1)
        self.decconv1=nn.Conv2d(32,out_channels,3,1,1)
        self.activ= self.out_activation

    def forward(self, x):
        # print('Decoder input range', x.min(),x.max())

        # plt.imsave(savepath+'dec_input.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray)

        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.convlayer4(x)
        x = self.decconv1(x)
        x = self.activ(x)

        # plt.imsave(savepath+'dec_output.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray)
        # print('decoder output range', x.min(),x.max())

        return x

@baseline_registry.register_obs_transformer()
class E2E_Decoder(ObservationTransformer):
    def __init__(self, in_channels=1, out_channels=1, out_activation='relu'):
        super().__init__()

        self.transformed_sensor = 'rgb'

        self.model = Decoder()

    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        observations['rgb'] = self._transform_obs(observations['rgb'])
        return observations

    def _transform_obs(self, observation: torch.Tensor):
        observation_sliced=observation.permute(0,3,1,2)[:,0,:,:].unsqueeze(1)

        # print('OBS_SHAPE',observation.shape, observation_sliced.shape)

        reconstruction = self.model.forward(observation_sliced)
        reconstruction = reconstruction.permute(0,2,3,1)

        return reconstruction

    @classmethod
    def from_config(cls, config: get_config):
        return cls()

def perlin_noise_map(seed=0,shape=(256,256),scale=100,octaves=6,persistence=.5,lacunarity=2.):
    out = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            out[i][j] = noise.pnoise2(i/scale,
                                        j/scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=shape[0],
                                        repeaty=shape[1],
                                        base=seed)
    out = (out-out.min())/(out.max()-out.min())
    return out

def get_pMask(size=(256,256),phosphene_density=32,seed=1,
              jitter_amplitude=0., intensity_var=0.,
              dropout=False,perlin_noise_scale=.4):

    # Define resolution and phosphene_density
    [nx,ny] = size
    n_phosphenes = phosphene_density**2 # e.g. n_phosphenes = 32 x 32 = 1024
    pMask = torch.zeros(size)

    # Custom 'dropout_map'
    p_dropout = perlin_noise_map(shape=size,scale=perlin_noise_scale*size[0],seed=seed)
    np.random.seed(seed)

    for p in range(n_phosphenes):
        i, j = divmod(p, phosphene_density)

        jitter = np.round(np.multiply(np.array([nx,ny])//phosphene_density,
                                      jitter_amplitude * (np.random.rand(2)-.5))).astype(int)
        rx = (j*nx//phosphene_density) + nx//(2*phosphene_density) + jitter[0]
        ry = (i*ny//phosphene_density) + ny//(2*phosphene_density) + jitter[1]

        rx = np.clip(rx,0,nx-1)
        ry = np.clip(ry,0,ny-1)

        intensity = intensity_var*(np.random.rand()-0.5)+1.
        if dropout==True:
            pMask[rx,ry] = np.random.choice([0.,intensity], p=[p_dropout[rx,ry],1-p_dropout[rx,ry]])
        else:
            pMask[rx,ry] = intensity

    return pMask

@baseline_registry.register_obs_transformer()
class E2E_PhospheneSimulator(ObservationTransformer):
    """ Uses three steps to convert  the stimulation vectors to phosphene representation:
    1. Resizes the feature map (default: 32x32) to SPV template (256x256)
    2. Uses pMask to sample the phosphene locations from the SPV activation template
    2. Performs convolution with gaussian kernel for realistic phosphene simulations
    """
    def __init__(self,scale_factor=8, sigma=1.5,kernel_size=11, intensity=15):
        super().__init__()

        # Phosphene grid
        self.pMask = get_pMask(jitter_amplitude=0,dropout=False,seed=0)
        self.up = nn.Upsample(mode="nearest",scale_factor=scale_factor)
        self.gaussian = self.get_gaussian_layer(kernel_size=kernel_size, sigma=sigma, channels=1)
        self.intensity = intensity

        # self.gaussian.to("cuda:0")
        # self.up.to("cuda:0")
        # self.pMask.to("cuda:0")

        self.transformed_sensor = 'rgb'


    def get_gaussian_layer(self, kernel_size, sigma, channels):
        """non-trainable Gaussian filter layer for more realistic phosphene simulation"""

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) /\
                              (2*variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter


    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        key = self.transformed_sensor
        if key in observations:
            observations[key] = self.transform_obs(observations[key])
        return observations

    def transform_obs(self, observation: torch.Tensor):
        # print('Sim input range', observation.permute(0,3,1,2).min(),observation.permute(0,3,1,2).max())

        # The sim_input torch.Size([64, 1, 32, 32])
        # The sim_output torch.Size([64, 1, 256, 256])

        # plt.imsave(savepath+'sim_input.png', observation.permute(0,3,1,2)[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray)
        device = observation.device

        phosphenes = self.up(observation.permute(0,3,1,2).float())*self.pMask
        phosphenes = self.gaussian(F.pad(phosphenes, (5,5,5,5), mode='constant', value=0))
        phosphene = self.intensity*phosphenes

        # print('Sim output range', phosphene.min(),phosphene.max())
        # plt.imsave(savepath+'sim_output.png', phosphene[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray)
        phosphene = torch.tile(phosphene.permute(0,2,3,1), (1,1,1,3))

        return phosphene

    @classmethod
    def from_config(cls, config: get_config):
        return cls()


def apply_E2E_Encoder(Img, checkpoint_folder, checkpoint_name):
    model = E2E_Encoder()

    imgGray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.resize(imgGray, (128, 128), interpolation=cv2.INTER_AREA)
    convert_tensor = transforms.ToTensor()
    img_tensor = convert_tensor(imgGray)

    checkpoint_dir = ("/scratch/big/home/carsan/Data/phosphenes/habitat/checkpoints/"+
                      checkpoint_folder+"/"+checkpoint_name)

    state_dict_encoder = torch.load(checkpoint_dir)["state_dict_encoder"]
    model.load_state_dict(
        {  # type: ignore
            k[len("model."):]: v
            for k, v in state_dict_encoder.items()
            if "model" in k
        }
    )

    # Apply to image
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.permute(0, 2, 3, 1)

    stimulation = model.forward(img_tensor.permute(0,3,1,2).float())
    # stimulation = .5*(frame+1)  # Is this important?
    stimulation = stimulation.permute(0, 2, 3, 1)

    return stimulation

def apply_phosphene_simulator(Img):
    # Load model
    simulator = E2E_PhospheneSimulator()

    # Apply tranformation
    phosphenes = simulator.transform_obs(Img)

    return phosphenes


def apply_decoder(phosphenes, checkpoint_folder, checkpoint_name):
    model = Decoder()

    checkpoint_dir = ("/scratch/big/home/carsan/Data/phosphenes/habitat/checkpoints/" +
                      checkpoint_folder + "/" + checkpoint_name)

    state_dict_encoder = torch.load(checkpoint_dir)["state_dict_decoder"]
    model.load_state_dict(
        {  # type: ignore
            k[len("model."):]: v
            for k, v in state_dict_encoder.items()
            if "model" in k
        }
    )

    observation_sliced = phosphenes.permute(0, 3, 1, 2)[:, 0, :, :].unsqueeze(1)

    reconstruction = model.forward(observation_sliced)
    reconstruction = reconstruction.permute(0, 2, 3, 1)

    return reconstruction

def plot_img(input, encoder, phosphenes, decoder, save_name):
    show_grid = False
    sns.set(style="whitegrid" if show_grid else "white")

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(top=0.85)

    ax[0][0].imshow(input, cmap='gray' if input.ndim == 2 else None)
    ax[0][0].set_title("Observation")
    ax[0][0].axis('off')
    ax[0][0].set_xticks([])  # Remove x-axis ticks
    ax[0][0].set_yticks([])  # Remove y-axis ticks

    encoderBig = cv2.resize(encoder.squeeze(), (32,32), 8, 8, cv2.INTER_LINEAR)
    ax[0][1].imshow(encoder, cmap='gray' if encoder.ndim == 2 else None)
    ax[0][1].set_title("Encoder")
    ax[0][1].axis('off')
    ax[0][1].set_xticks([])  # Remove x-axis ticks
    ax[0][1].set_yticks([])  # Remove y-axis ticks

    ax[1][0].imshow(phosphenes, cmap='gray' if phosphenes.ndim == 2 else None)
    ax[1][0].set_title("Phosphenes")
    ax[1][0].axis('off')
    ax[1][0].set_xticks([])  # Remove x-axis ticks
    ax[1][0].set_yticks([])  # Remove y-axis ticks

    ax[1][1].imshow(decoder, cmap='gray' if input.ndim == 2 else None)
    ax[1][1].set_title("Decoder")
    ax[1][1].axis('off')
    ax[1][1].set_xticks([])  # Remove x-axis ticks
    ax[1][1].set_yticks([])  # Remove y-axis ticks

    save_dir = '/scratch/big/home/carsan/Internship/PyCharm_projects/habitat_2.3/habitat-phosphenes/manualPhospheneRepresentations'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'{save_name}.png')
    # Set the general title for the figure
    fig.suptitle("Phosphene Vision via E2E-optimization", fontsize=16)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# Starts as (6, 128, 128, 1) at beginning of decoder
# observation.permute(0,3,1,2) changes to (6, 128, 128, 1)
# After encoder is (6, 1, 32, 32)
# After permute it comes back to (6, 32, 32, 1) this enters to simulator

# First permute changes it to (6, 1, 32, 32)
# Phosphenes starts as (6, 1, 256, 256)
# Phosphene ends as (6, 256, 256, 3)

# In decoder starts (96, 256, 256, 3) 96 coming from num_steps x (environments/numMiniBatch)
# Obs sliced goes to (96, 1, 256, 256)
# Reconstruction (96, 1, 128, 128)
# Final reconstruction (96, 128, 128, 1)


if __name__ == "__main__":
    # Params
    checkpoint_folder = "train_NPT_RGB_E2E_2.8Msteps_2Envs_800spe_xavierWeights_lr=e-4_32spu_weightLossPPO=0.6_decoderWithRelu"
    checkpoint_name = "ckpt_phos.24.pth"
    img_dir = "/scratch/big/home/carsan/Data/habitatai/images/bin/obs_input_20240104_235720.png"
    save_name_for_img = "trial1"

    environment = cv2.imread(img_dir)

    img_encoder = apply_E2E_Encoder(environment, checkpoint_folder, checkpoint_name)

    phosphenes = apply_phosphene_simulator(img_encoder)

    img_decoder = apply_decoder(phosphenes, checkpoint_folder, checkpoint_name)

    plot_img(environment.squeeze(), img_encoder.detach().numpy().squeeze(), phosphenes.detach().numpy().squeeze(), img_decoder.detach().numpy().squeeze(), save_name_for_img)

    print("END")











