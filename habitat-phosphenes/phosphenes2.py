import copy
from typing import Dict

import cv2
import numpy as np
from dataclasses import dataclass

import torch
from gym.spaces import Box

import torch.nn as nn
import torch.nn.functional as F
import math
import noise

#added
from PIL import Image
import matplotlib.pyplot as plt

from habitat import Config
from habitat.core import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.utils.common import get_image_height_width

#added
# savepath= '/home/burkuc/data/habitatai/images/'
savepath= '/home/burkuc/data/habitatai/images/bin/'

def overwrite_gym_box_shape(box: Box, shape) -> Box:
    if box.shape == shape:
        return box
    low = box.low if np.isscalar(box.low) else np.min(box.low)
    high = box.high if np.isscalar(box.high) else np.max(box.high)
    return Box(low=low, high=high, shape=shape, dtype=box.dtype)


@baseline_registry.register_obs_transformer()
class GrayScale(ObservationTransformer):
    def __init__(self): #, resize_needed):
        super().__init__()
        self.transformed_sensor = 'rgb'

        # self.resize_needed = resize_needed

    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        if self.transformed_sensor in observations:
            observations[self.transformed_sensor] = self._transform_obs(
                observations[self.transformed_sensor])
        return observations

    @staticmethod
    def _transform_obs(observation):
        device = observation.device

        # img=Image.fromarray(observation[0,:,:,:].detach().cpu().numpy().astype(np.uint8))
        # img.save('/home/burkuc/data/habitatai/images/gray_input.png')

        observation = observation.cpu().numpy()
        

        frames = []
        for frame in observation:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # if self.resize_needed:
                #added if resized needed for e2e pipeline
            frame = cv2.resize(frame, (128,128), interpolation=cv2.INTER_AREA)

            frames.append(frame)

        # print('frame gray', frame.shape) #(256, 256)
        # print('frames', frames)
        observation = torch.as_tensor(np.expand_dims(frames, -1),
                                      device=device)

        # print('observation gray', observation.shape) #torch.Size([1, 256, 256, 1])   
        # img=Image.fromarray(observation[0,:,:,0].detach().cpu().numpy().astype(np.uint8))
        # img.save('/home/burkuc/data/habitatai/images/gray_output_prenorm.png')


        # print("grayscale",observation.min(), observation.max())

        observation = (
                observation.float() / 255.0
            ) 

        # print("grayscale after normalize",observation.min(), observation.max())

        # img=Image.fromarray(observation[0,:,:,0].detach().cpu().numpy().astype(np.uint8))
        # img.save('/home/burkuc/data/habitatai/images/gray_norm_output.png') # black
        # img=Image.fromarray(observation[0,:,:,0].detach().cpu().numpy())
        # img.save('/home/burkuc/data/habitatai/images/gray_norm_output_float.png') #doesnt save give error
        # plt.imsave('/home/burkuc/data/habitatai/images/gray_norm_output_float_imsave.png', observation[0,:,:,0].detach().cpu().numpy()) #purply colors
        # plt.imsave('/home/burkuc/data/habitatai/images/gray_norm_output_float_imsave_cmap.png', observation[0,:,:,0].detach().cpu().numpy(), cmap=plt.cm.gray) #desired
        # plt.savefig('/home/burkuc/data/habitatai/images/gray_norm_output_float_savefig.png') #all white
        plt.imsave(savepath+'gray_output.png', observation[0,:,:,0].detach().cpu().numpy(), cmap=plt.cm.gray)

        return observation

    def transform_observation_space(self, observation_space: spaces.Dict,
                                    **kwargs):
        key = self.transformed_sensor
        observation_space = copy.deepcopy(observation_space)

        h, w = get_image_height_width(observation_space[key],
                                      channels_last=True)
        new_shape = (h, w, 1)
        observation_space[key] = overwrite_gym_box_shape(
            observation_space[key], new_shape)
        return observation_space

    @classmethod
    def from_config(cls, config: Config):
        return cls()


@baseline_registry.register_obs_transformer()
class EdgeFilter(ObservationTransformer):

    def __init__(self, sigma, threshold_low, threshold_high):
        super().__init__()
        self.sigma = sigma
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.transformed_sensor = 'rgb'

    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        key = self.transformed_sensor
        if key in observations:
            observations[key] = self._transform_obs(observations[key])
        return observations

    def _transform_obs(self, observation: torch.Tensor):
        device = observation.device

        observation = observation.cpu().numpy()

        frames = []
        for frame in observation:

            # Gaussian blur to remove noise.
            frame = cv2.GaussianBlur(frame, ksize=None, sigmaX=self.sigma)

            # Canny edge detection.
            frame = cv2.Canny(frame, self.threshold_low, self.threshold_high)

            # Copy grayscale image on each RGB channel so we can reuse
            # pre-trained net.
            # frames.append(np.tile(np.expand_dims(frame, -1), 3))
            frames.append(np.expand_dims(frame, -1))
            # print('frame edge', frame.shape) #(256, 256) 
            # print('frames', frames)

        observation = torch.as_tensor(np.array(frames), device=device)
        # print('edge obs', observation.shape) #torch.Size([1, 256, 256, 1])

        return observation

    @classmethod
    def from_config(cls, config: Config):
        c = config.RL.POLICY.OBS_TRANSFORMS.EDGE_FILTER
        return cls(c.SIGMA, c.THRESHOLD_LOW, c.THRESHOLD_HIGH)


@baseline_registry.register_obs_transformer()
class Phosphenes(ObservationTransformer):
    def __init__(self, size, phosphene_resolution, sigma):
        super().__init__()
        self.sigma = sigma
        jitter = 0.4
        intensity_var = 0.8
        aperture = 0.66
        self.transformed_sensor = 'rgb'
        self.grid = create_regular_grid((phosphene_resolution,
                                         phosphene_resolution),
                                        size, jitter, intensity_var)
        # relative aperture > dilation kernel size
        aperture = np.round(aperture *
                            size[0] / phosphene_resolution).astype(int)
        self.dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                         (aperture, aperture))

    @classmethod
    def from_config(cls, config: Config):
        c = config.RL.POLICY.OBS_TRANSFORMS.PHOSPHENES
        return cls(c.SIZE, c.RESOLUTION, c.SIGMA)

    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        key = self.transformed_sensor
        if key in observations:
            observations[key] = self._transform_obs(observations[key])
        return observations

    def _transform_obs(self, observation):
        device = observation.device

        observation = observation.cpu().numpy()

        frames = []
        for frame in observation:
            mask = cv2.dilate(frame, self.dilation_kernel, iterations=1)
            phosphenes = self.grid * mask
            phosphenes = cv2.GaussianBlur(phosphenes, ksize=None,
                                          sigmaX=self.sigma)
            # Copy grayscale image on each RGB channel so we can reuse
            # pre-trained net.
            phosphenes = 255 * phosphenes / (phosphenes.max() or 1)
            frames.append(np.tile(np.expand_dims(phosphenes, -1), 3))

        # print('frames phos', phosphenes.shape) #(256, 256) 
        # print('frames', frames)
        phosphenes = torch.as_tensor(np.array(frames, 'uint8'), device=device)
        # print('observation phos', phosphenes.shape) #torch.Size([1, 256, 256, 3])
        return phosphenes

    def transform_observation_space(self, observation_space: spaces.Dict,
                                    **kwargs):
        key = self.transformed_sensor
        observation_space = copy.deepcopy(observation_space)

        h, w = get_image_height_width(observation_space[key],
                                      channels_last=True)
        new_shape = (h, w, 3)
        observation_space[key] = overwrite_gym_box_shape(
            observation_space[key], new_shape)
        return observation_space


def create_regular_grid(phosphene_resolution, size, jitter, intensity_var):
    """Returns regular eqiodistant phosphene grid of shape <size> with
    resolution <phosphene_resolution> for variable phosphene intensity with
    jittered positions"""

    grid = np.zeros(size)
    phosphene_spacing = np.divide(size, phosphene_resolution)
    xrange = np.linspace(0, size[0], phosphene_resolution[0], endpoint=False) \
        + phosphene_spacing[0] / 2
    yrange = np.linspace(0, size[1], phosphene_resolution[1], endpoint=False) \
        + phosphene_spacing[1] / 2
    for x in xrange:
        for y in yrange:
            deviation = \
                jitter * (2 * np.random.rand(2) - 1) * phosphene_spacing
            intensity = intensity_var * (np.random.rand() - 0.5) + 1
            rx = \
                np.clip(np.round(x + deviation[0]), 0, size[0] - 1).astype(int)
            ry = \
                np.clip(np.round(y + deviation[1]), 0, size[1] - 1).astype(int)
            grid[rx, ry] = intensity
    return grid

#todo: make new classes in pytorch! 
# for encoder
# for simulator if needed 
# (biologic simulator later if proof with the first one works),

#beware _transform_obs ​is where the main change happens in the classes  

def convlayer(n_input, n_output, k_size=3, stride=1, padding=1, resample_out=None):
    layer = [
        nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(n_output),
        nn.LeakyReLU(inplace=False), #True
        resample_out]
    if resample_out is None:
        layer.pop()
    return layer  

class ConvLayer(nn.Module):
    def __init__(self, n_input, n_output,  k_size=3, stride=1, padding=1): 
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=n_input, out_channels=n_output,  kernel_size=k_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(n_output)
        self.leaky = nn.LeakyReLU(inplace=False) #True

    def forward(self, x):
        out = self.leaky(self.bn(self.conv(x)))
        return out

class ConvLayer2(nn.Module):
    def __init__(self, n_input, n_output,  k_size=3, stride=1, padding=1): 
        super(ConvLayer2, self).__init__()

        self.conv = nn.Conv2d(in_channels=n_input, out_channels=n_output,  kernel_size=k_size, stride=stride, padding=padding, bias=False)
        self.swish = nn.SiLU() #nn.Swish() 

    def forward(self, x):
        out = self.swish(self.conv(x))
        return out

class ResidualBlock(nn.Module):
    def __init__(self, n_channels, stride=1, resample_out=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu = nn.LeakyReLU(inplace=False) #True
        self.conv2 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.resample_out = resample_out
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        if self.resample_out:
            out = self.resample_out(out)
        return out

class ResidualBlock2(nn.Module):
    def __init__(self, n_channels, stride=1, resample_out=None):
        super(ResidualBlock2, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.swish = nn.SiLU() #nn.Swish()
        self.conv2 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        
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
        
        # Model
                #jaap
        # self.model = nn.Sequential(ConvLayer(in_channels,8,3,1,1), # *convlayer(in_channels,8,3,1,1),
        #                     ConvLayer(8,16,3,1,1),
        #                     nn.MaxPool2d(2), #*convlayer(8,16,3,1,1,resample_out=nn.MaxPool2d(2)),
        #                     ConvLayer(16,32,3,1,1),
        #                     nn.MaxPool2d(2), #*convlayer(16,32,3,1,1,resample_out=nn.MaxPool2d(2)),
        #                     ResidualBlock(32, resample_out=None),
        #                     ResidualBlock(32, resample_out=None),
        #                     ResidualBlock(32, resample_out=None),
        #                     ResidualBlock(32, resample_out=None),
        #                     ConvLayer(32,16,3,1,1),
        #                     nn.Conv2d(16,out_channels,3,1,1),  
        #                     nn.Tanh())
        
        #tophos
        # self.model = nn.Sequential(ConvLayer2(in_channels,8,3,1,1), # *convlayer(in_channels,8,3,1,1),
        #                     ConvLayer2(8,16,3,1,1),
        #                     nn.MaxPool2d(2), #*convlayer(8,16,3,1,1,resample_out=nn.MaxPool2d(2)),
        #                     ConvLayer2(16,32,3,1,1),
        #                     nn.MaxPool2d(2), #*convlayer(16,32,3,1,1,resample_out=nn.MaxPool2d(2)),
        #                     ResidualBlock2(32, resample_out=None),
        #                     ResidualBlock2(32, resample_out=None),
        #                     ResidualBlock2(32, resample_out=None),
        #                     ResidualBlock2(32, resample_out=None),
        #                     ConvLayer2(32,16,3,1,1),
        #                     nn.Conv2d(16,out_channels,3,1,1),  
        #                     nn.Tanh())

        self.convlayer1 = ConvLayer2(in_channels,8,3,1,1) 
        self.convlayer2 = ConvLayer2(8,16,3,1,1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.convlayer3 = ConvLayer2(16,32,3,1,1)
        self.maxpool2 =nn.MaxPool2d(2)
        self.res1 = ResidualBlock2(32)
        self.res2 = ResidualBlock2(32)
        self.res3 = ResidualBlock2(32)
        self.res4 = ResidualBlock2(32)
        self.convlayer4 =ConvLayer2(32,16,3,1,1)
        self.encconv1 = nn.Conv2d(16,out_channels,3,1,1) #bias true
        self.tanh1 = nn.Tanh()

    def forward(self, x):
        # self.out = self.model(x)
        # x = self.out
        # if self.binary_stimulation:
        #     x = x + torch.sign(x).detach() - x.detach() # (self-through estimator)

        print('encoder input range', x.min(),x.max(), x.shape)

        # print('enc_input',x.shape)

        # enc_input torch.Size([64, 1, 128, 128])                                                                            
        # enc_convlayer1 torch.Size([64, 8, 128, 128])                                                                       
        # enc_convlayer2 torch.Size([64, 16, 64, 64])                                                                        
        # enc_convlayer3 torch.Size([64, 32, 32, 32])                                                                        
        # enc_res1 torch.Size([64, 32, 32, 32])                                                                              
        # enc_res2 torch.Size([64, 32, 32, 32])                                                                              
        # enc_res3 torch.Size([64, 32, 32, 32])                                                                              
        # enc_res4 torch.Size([64, 32, 32, 32])
        # enc_convlayer4 torch.Size([64, 16, 32, 32])
        # enc_tanh torch.Size([64, 1, 32, 32])
        # enc_output torch.Size([64, 1, 32, 32])

        # plt.imsave(savepath+'enc_input.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        x = self.convlayer1(x)
        # print('enc_convlayer1',x.shape)
        # plt.imsave(savepath+'enc_convlayer1.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        x = self.maxpool1(self.convlayer2(x))
        # print('enc_convlayer2',x.shape)
        # plt.imsave(savepath+'enc_convlayer2.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        x = self.maxpool2(self.convlayer3(x))
        # print('enc_convlayer3',x.shape)
        # plt.imsave(savepath+'enc_convlayer3.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        x = self.res1(x)
        # print('enc_res1',x.shape)
        # plt.imsave(savepath+'enc_res1.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        x = self.res2(x)
        # print('enc_res2',x.shape)
        # plt.imsave(savepath+'enc_res2.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        x = self.res3(x)
        # print('enc_res3',x.shape)
        # plt.imsave(savepath+'enc_res3.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        x = self.res4(x)
        # print('enc_res4',x.shape)
        # plt.imsave(savepath+'enc_res4.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        x = self.convlayer4(x)
        # print('enc_convlayer4',x.shape)
        # plt.imsave(savepath+'enc_convlayer4.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        x = self.tanh1(self.encconv1(x))
        # print('enc_tanh',x.shape)
        # plt.imsave(savepath+'enc_tanh.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 

        # if self.binary_stimulation:
        #    x = x + torch.sign(x).detach() - x.detach() # (self-through estimator)

        stimulation = .5*(x+1)
        # print('enc_output',stimulation.shape)
        # img=Image.fromarray(stimulation[0,0,:,:].detach().cpu().numpy().astype(np.uint8))
        # img.save(savepath+'enc_output.png')
        # plt.imsave('/home/burkuc/data/habitatai/images/enc_output_float_imsave_cmap.png', stimulation[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        plt.imsave(savepath+'enc_output.png', stimulation[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        return stimulation 

@baseline_registry.register_obs_transformer()
class Encoder(ObservationTransformer):
    # model: nn.Module

    # def __init__(self, in_channels=3, out_channels=1,binary_stimulation=True):
    def __init__(self, in_channels=1, out_channels=1,binary_stimulation=True):
        super().__init__()

        self.binary_stimulation = binary_stimulation

        self.transformed_sensor = 'rgb'

        # print('inchannels', in_channels)
        #jaap
        # self.model = nn.Sequential(ConvLayer(in_channels,8,3,1,1), # *convlayer(in_channels,8,3,1,1),
        #                     ConvLayer(8,16,3,1,1),
        #                     nn.MaxPool2d(2), #*convlayer(8,16,3,1,1,resample_out=nn.MaxPool2d(2)),
        #                     ConvLayer(16,32,3,1,1),
        #                     nn.MaxPool2d(2), #*convlayer(16,32,3,1,1,resample_out=nn.MaxPool2d(2)),
        #                     ResidualBlock(32, resample_out=None),
        #                     ResidualBlock(32, resample_out=None),
        #                     ResidualBlock(32, resample_out=None),
        #                     ResidualBlock(32, resample_out=None),
        #                     ConvLayer(32,16,3,1,1),
        #                     nn.Conv2d(16,out_channels,3,1,1),  
        #                     nn.Tanh())
        
        #tophos
        self.model = E2E_Encoder()
        # self.model = nn.Sequential(ConvLayer2(in_channels,8,3,1,1), # *convlayer(in_channels,8,3,1,1),
        #                     ConvLayer2(8,16,3,1,1),
        #                     nn.MaxPool2d(2), #*convlayer(8,16,3,1,1,resample_out=nn.MaxPool2d(2)),
        #                     ConvLayer2(16,32,3,1,1),
        #                     nn.MaxPool2d(2), #*convlayer(16,32,3,1,1,resample_out=nn.MaxPool2d(2)),
        #                     ResidualBlock2(32, resample_out=None),
        #                     ResidualBlock2(32, resample_out=None),
        #                     ResidualBlock2(32, resample_out=None),
        #                     ResidualBlock2(32, resample_out=None),
        #                     ConvLayer2(32,16,3,1,1),
        #                     nn.Conv2d(16,out_channels,3,1,1),  
        #                     nn.Tanh())
        # self.train()

    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        key = self.transformed_sensor
        if key in observations:
            observations[key] = self._transform_obs(observations[key])
        return observations

    def _transform_obs(self, observation: torch.Tensor):
        # print('input of encoder observationshape', observation.shape)

        # device = observation.device
        # self.model.cuda(device)
        
        # print('permuted', observation.permute(0,3,1,2).shape)
        stimulation = self.model.forward(observation.permute(0,3,1,2).float())

        # self.out = self.model.forward(observation.permute(0,3,1,2).float())
        # frame = self.out
        # #if japp
        # #if self.binary_stimulation:
        # #    frame = frame + torch.sign(frame).detach() - frame.detach() # (self-through estimator)
        # stimulation = .5*(frame+1)
        stimulation = stimulation.permute(0,2,3,1)

        # print('stimulation',stimulation.shape)

        return stimulation
    
    @classmethod
    def from_config(cls, config: Config):
        return cls()
    
    # @classmethod
    # def from_config(cls, config: Config):
    #     c = config.RL.POLICY.OBS_TRANSFORMS.PHOSPHENES
    #     return cls(c.SIZE, c.RESOLUTION, c.SIGMA)
    
    def transform_observation_space(self, observation_space: spaces.Dict,
                                    **kwargs):
        key = self.transformed_sensor
        observation_space = copy.deepcopy(observation_space)

        h, w = get_image_height_width(observation_space[key],
                                      channels_last=True)
        new_shape = (h, w, 1)
        observation_space[key] = overwrite_gym_box_shape(
            observation_space[key], new_shape)
        return observation_space

class Decoder(nn.Module):
    """
    Simple non-generic phosphene decoder.
    in: (256x256) SVP representation
    out: (128x128) Reconstruction
    """   
    def __init__(self, in_channels=1, out_channels=1, out_activation='sigmoid'):
        super(Decoder, self).__init__()
             
        # Activation of output layer
        self.out_activation = {'tanh': nn.Tanh(),
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.LeakyReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]
        
        # # Model
        # self.model = nn.Sequential(ConvLayer(in_channels,16,3,1,1), #*convlayer(in_channels,16,3,1,1),
        #                            ConvLayer(16,32,3,1,1),#*convlayer(16,32,3,1,1),
        #                            ConvLayer(32,64,3,2,1), #*convlayer(32,64,3,2,1),
        #                            ResidualBlock(64),
        #                            ResidualBlock(64),
        #                            ResidualBlock(64),
        #                            ResidualBlock(64),
        #                            ConvLayer(64,32,3,1,1), #*convlayer(64,32,3,1,1),
        #                            nn.Conv2d(32,out_channels,3,1,1), 
        #                            self.out_activation)     
        
        # self.model = nn.Sequential(ConvLayer2(in_channels,16,3,1,1), #*convlayer(in_channels,16,3,1,1),
        #                            ConvLayer2(16,32,3,1,1),#*convlayer(16,32,3,1,1),
        #                            ConvLayer2(32,64,3,2,1), #*convlayer(32,64,3,2,1),
        #                            ResidualBlock2(64),
        #                            ResidualBlock2(64),
        #                            ResidualBlock2(64),
        #                            ResidualBlock2(64),
        #                            ConvLayer2(64,32,3,1,1), #*convlayer(64,32,3,1,1),
        #                            nn.Conv2d(32,out_channels,3,1,1), 
        #                            self.out_activation)    

        self.convlayer1=ConvLayer2(in_channels,16,3,1,1)
        self.convlayer2=ConvLayer2(16,32,3,1,1)
        self.convlayer3=ConvLayer2(32,64,3,2,1)
        self.res1=ResidualBlock2(64)
        self.res2=ResidualBlock2(64)
        self.res3=ResidualBlock2(64)
        self.res4=ResidualBlock2(64)
        self.convlayer4=ConvLayer2(64,32,3,1,1)
        self.decconv1=nn.Conv2d(32,out_channels,3,1,1)
        self.activ= self.out_activation 
    
    def forward(self, x):
        # print('dec_input', x.shape)
        print('decoder input range', x.min(),x.max())

        # dec_input torch.Size([64, 1, 256, 256])
        # dec_convlayer1 torch.Size([64, 16, 256, 256])
        # dec_convlayer2 torch.Size([64, 32, 256, 256])
        # dec_convlayer3 torch.Size([64, 64, 128, 128])
        # dec_res1 torch.Size([64, 64, 128, 128])
        # dec_res2 torch.Size([64, 64, 128, 128])
        # dec_res3 torch.Size([64, 64, 128, 128])
        # dec_res4 torch.Size([64, 64, 128, 128])
        # dec_convlayer4 torch.Size([64, 32, 128, 128])
        # dec_convlayer5 torch.Size([64, 1, 128, 128])
        # dec_output torch.Size([64, 1, 128, 128])

        # plt.imsave(savepath+'dec_input.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        x = self.convlayer1(x)
        # print('dec_convlayer1', x.shape)
        # plt.imsave(savepath+'dec_convlayer1.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        x = self.convlayer2(x)
        # print('dec_convlayer2', x.shape)
        # plt.imsave(savepath+'dec_convlayer2.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        x = self.convlayer3(x)
        # print('dec_convlayer3', x.shape)
        # plt.imsave(savepath+'dec_convlayer3.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        x = self.res1(x)
        # print('dec_res1', x.shape)
        # plt.imsave(savepath+'dec_res1.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        x = self.res2(x)
        # print('dec_res2', x.shape)
        # plt.imsave(savepath+'dec_res2.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        x = self.res3(x)
        # print('dec_res3', x.shape)
        # plt.imsave(savepath+'dec_res3.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        x = self.res4(x)
        # print('dec_res4', x.shape)
        # plt.imsave(savepath+'dec_res4.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        x = self.convlayer4(x)
        # print('dec_convlayer4', x.shape)
        # plt.imsave(savepath+'dec_convlayer4.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        x = self.decconv1(x)
        # print('dec_convlayer5', x.shape)
        # plt.imsave(savepath+'dec_convlayer5.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        x = self.activ(x)
        # print('dec_output', x.shape)
        plt.imsave(savepath+'dec_output.png', x[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 

        print('decoder output range', x.min(),x.max())

        return x
        # return self.model(x)

@baseline_registry.register_obs_transformer()
class E2E_Decoder(ObservationTransformer):
    # model: nn.Module

    def __init__(self, in_channels=1, out_channels=1, out_activation='sigmoid'):
        super().__init__()

        self.transformed_sensor = 'rgb'

        # Activation of output layer
        self.out_activation = {'tanh': nn.Tanh(),
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.LeakyReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]
        
        # # Model
        # self.model = nn.Sequential(ConvLayer(in_channels,16,3,1,1), #*convlayer(in_channels,16,3,1,1),
        #                            ConvLayer(16,32,3,1,1),#*convlayer(16,32,3,1,1),
        #                            ConvLayer(32,64,3,2,1), #*convlayer(32,64,3,2,1),
        #                            ResidualBlock(64),
        #                            ResidualBlock(64),
        #                            ResidualBlock(64),
        #                            ResidualBlock(64),
        #                            ConvLayer(64,32,3,1,1), #*convlayer(64,32,3,1,1),
        #                            nn.Conv2d(32,out_channels,3,1,1), 
        #                            self.out_activation)     
        
        self.model = Decoder() 
        # self.model =  nn.Sequential(ConvLayer2(in_channels,16,3,1,1), #*convlayer(in_channels,16,3,1,1),
        #                            ConvLayer2(16,32,3,1,1),#*convlayer(16,32,3,1,1),
        #                            ConvLayer2(32,64,3,2,1), #*convlayer(32,64,3,2,1),
        #                            ResidualBlock2(64),
        #                            ResidualBlock2(64),
        #                            ResidualBlock2(64),
        #                            ResidualBlock2(64),
        #                            ConvLayer2(64,32,3,1,1), #*convlayer(64,32,3,1,1),
        #                            nn.Conv2d(32,out_channels,3,1,1), 
        #                            self.out_activation)     
        # self.train()
        
    
    def forward(self, observations: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        # key = self.transformed_sensor
        # if key in observations:
            # observations[key] = self._transform_obs(observations[key])
        observations['rgb'] = self._transform_obs(observations['rgb'])
        return observations

    def _transform_obs(self, observation: torch.Tensor):
        # print('input of encoder observationshape', observation.shape)

        # device = observation.device
        # self.model.cuda(device)
        observation_sliced=observation.permute(0,3,1,2)[:,0,:,:].unsqueeze(1)
        print('OBSSHAPE',observation.shape, observation_sliced.shape)

        reconstruction = self.model.forward(observation_sliced)
        reconstruction = reconstruction.permute(0,2,3,1)
        # print('stimulation',stimulation.shape)

        return reconstruction

    @classmethod
    def from_config(cls, config: Config):
        return cls()
    
    # @classmethod
    # def from_config(cls, config: Config):
    #     c = config.RL.POLICY.OBS_TRANSFORMS.PHOSPHENES
    #     return cls(c.SIZE, c.RESOLUTION, c.SIGMA)
    
    def transform_observation_space(self, observation_space: spaces.Dict,
                                    **kwargs):
        key = self.transformed_sensor
        observation_space = copy.deepcopy(observation_space)

        h, w = get_image_height_width(observation_space[key],
                                      channels_last=True)
        new_shape = (h, w, 1)
        observation_space[key] = overwrite_gym_box_shape(
            observation_space[key], new_shape)
        return observation_space



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
    1. Resizes the feature map (default: 32x32) to SVP template (256x256)
    2. Uses pMask to sample the phosphene locations from the SVP activation template
    2. Performs convolution with gaussian kernel for realistic phosphene simulations
    """
    # def __init__(self,pMask,scale_factor=8, sigma=1.5,kernel_size=11, intensity=15):
    def __init__(self,scale_factor=8, sigma=1.5,kernel_size=11, intensity=15):
        super().__init__()
        
        # Phosphene grid
        self.pMask = get_pMask(jitter_amplitude=0,dropout=False,seed=0)
        self.up = nn.Upsample(mode="nearest",scale_factor=scale_factor)
        self.gaussian = self.get_gaussian_layer(kernel_size=kernel_size, sigma=sigma, channels=1)
        self.intensity = intensity 

        self.transformed_sensor = 'rgb'

        # self.train()
    
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
            observations[key] = self._transform_obs(observations[key])
        return observations

    def _transform_obs(self, observation: torch.Tensor):
        # print('input of simulator observationshape', observation.shape)
        device = observation.device
        # self.gaussian.cuda(device)

        print('sim input range', observation.permute(0,3,1,2).min(),observation.permute(0,3,1,2).max())

        # print('sim_input', observation.permute(0,3,1,2).shape)

        # sim_input torch.Size([64, 1, 32, 32])
        # sim_output torch.Size([64, 1, 256, 256])

        # plt.imsave(savepath+'sim_input.png', observation.permute(0,3,1,2)[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        
        phosphenes = self.up(observation.permute(0,3,1,2).float())*self.pMask.cuda(device)
        phosphenes = self.gaussian(F.pad(phosphenes, (5,5,5,5), mode='constant', value=0))
        phosphene = self.intensity*phosphenes 
        # print('sim_output', phosphene.shape)
        # img=Image.fromarray(phosphene[0,0,:,:].detach().cpu().numpy().astype(np.uint8))
        # img.save(savepath+'sim_output.png')
        # plt.imsave('/home/burkuc/data/habitatai/images/sim_output_float_imsave_cmap.png', phosphene[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 

        # print('sim output range', phosphene.min(),phosphene.max())

        plt.imsave(savepath+'sim_output.png', phosphene[0,0,:,:].detach().cpu().numpy(), cmap=plt.cm.gray) 
        phosphene = torch.tile(phosphene.permute(0,2,3,1), (1,1,1,3))
        # print('phosphene',phosphene.shape)
        
        return phosphene 

    @classmethod
    def from_config(cls, config: Config):
        return cls()
    
    
    def transform_observation_space(self, observation_space: spaces.Dict,
                                    **kwargs):
        key = self.transformed_sensor
        observation_space = copy.deepcopy(observation_space)

        h, w = get_image_height_width(observation_space[key],
                                      channels_last=True)
        new_shape = (h, w, 3)
        observation_space[key] = overwrite_gym_box_shape(
            observation_space[key], new_shape)
        return observation_space

class CustomLoss(object):
    def __init__(self, recon_loss_type='mse',recon_loss_param=None, stimu_loss_type=None, kappa=0): 

        """Custom loss class for training end-to-end model with a combination of reconstruction loss and sparsity loss
        reconstruction loss type can be either one of: 'mse' (pixel-intensity based), 'vgg' (i.e. perceptual loss/feature loss) 
        or 'boundary' (weighted cross-entropy loss on the output<>semantic boundary labels).
        stimulation loss type (i.e. sparsity loss) can be either 'L1', 'L2' or None.
        """
        
        # Reconstruction loss
        if recon_loss_type == 'mse':
            self.recon_loss = torch.nn.MSELoss()
            self.target = 'image'
        # elif recon_loss_type == 'vgg':
        #     self.feature_extractor = model.VGG_Feature_Extractor(layer_depth=recon_loss_param,device=device)
        #     self.recon_loss = lambda x,y: torch.nn.functional.mse_loss(self.feature_extractor(x),self.feature_extractor(y))
        #     self.target = 'image'
        # elif recon_loss_type == 'boundary':
        #     loss_weights = torch.tensor([1-recon_loss_param,recon_loss_param],device=device)
        #     self.recon_loss = torch.nn.CrossEntropyLoss(weight=loss_weights)
        #     self.target = 'label'

        # Stimulation loss 
        if stimu_loss_type=='L1':
            self.stimu_loss = lambda x: torch.mean(.5*(x+1)) #torch.mean(.5*(x+1)) #converts tanh to sigmoid first
        elif stimu_loss_type == 'L2':
            self.stimu_loss = lambda x: torch.mean((.5*(x+1))**2)  #torch.mean((.5*(x+1))**2) #converts tanh to sigmoid first
        elif stimu_loss_type is None:
            self.stimu_loss = None
        self.kappa = kappa if self.stimu_loss is not None else 0

    def prepare(self,observations):
        #get tensor from dicionary
        key = 'rgb'
        if key in observations:
            observations = observations[key]
        return observations
        
    def __call__(self,image,stimulation,phosphenes,reconstruction,validation=False):    
        device = image.device #self.prepare(image).device
        
        
        # Target
        if self.target == 'image': # Flag for reconstructing input image or target label
            # target = self.prepare(image)
            target = image
        # elif self.target == 'label':
        #     target = labelx
        
        # Calculate loss
        # loss_stimu = self.stimu_loss(self.prepare(stimulation)) if self.stimu_loss is not None else torch.zeros(1,device=device)
        loss_stimu = self.stimu_loss(stimulation) if self.stimu_loss is not None else torch.zeros(1,device=device)

        loss_recon = self.recon_loss(self.prepare(reconstruction),target)
        # print(reconstruction)
        # print(target)
        # print(loss_recon)
        # exit()

        loss_total = (1-self.kappa)*loss_recon + self.kappa*loss_stimu
        return loss_total, torch.mean(loss_recon), loss_stimu

# PREVIOUS BELOW< ABOVE IS CHANGED TO ASSUME THAT IMAGE ETC ARE NOT DICTIONARIES
# class CustomLoss(object):
#     def __init__(self, recon_loss_type='mse',recon_loss_param=None, stimu_loss_type=None, kappa=0): 

#         """Custom loss class for training end-to-end model with a combination of reconstruction loss and sparsity loss
#         reconstruction loss type can be either one of: 'mse' (pixel-intensity based), 'vgg' (i.e. perceptual loss/feature loss) 
#         or 'boundary' (weighted cross-entropy loss on the output<>semantic boundary labels).
#         stimulation loss type (i.e. sparsity loss) can be either 'L1', 'L2' or None.
#         """
        
#         # Reconstruction loss
#         if recon_loss_type == 'mse':
#             self.recon_loss = torch.nn.MSELoss()
#             self.target = 'image'
#         # elif recon_loss_type == 'vgg':
#         #     self.feature_extractor = model.VGG_Feature_Extractor(layer_depth=recon_loss_param,device=device)
#         #     self.recon_loss = lambda x,y: torch.nn.functional.mse_loss(self.feature_extractor(x),self.feature_extractor(y))
#         #     self.target = 'image'
#         # elif recon_loss_type == 'boundary':
#         #     loss_weights = torch.tensor([1-recon_loss_param,recon_loss_param],device=device)
#         #     self.recon_loss = torch.nn.CrossEntropyLoss(weight=loss_weights)
#         #     self.target = 'label'

#         # Stimulation loss 
#         if stimu_loss_type=='L1':
#             self.stimu_loss = lambda x: torch.mean(.5*(x+1)) #torch.mean(.5*(x+1)) #converts tanh to sigmoid first
#         elif stimu_loss_type == 'L2':
#             self.stimu_loss = lambda x: torch.mean((.5*(x+1))**2)  #torch.mean((.5*(x+1))**2) #converts tanh to sigmoid first
#         elif stimu_loss_type is None:
#             self.stimu_loss = None
#         self.kappa = kappa if self.stimu_loss is not None else 0

#     def prepare(self,observations):
#         #get tensor from dicionary
#         key = 'rgb'
#         if key in observations:
#             observations = observations[key]
#         return observations
        
#     def __call__(self,image,stimulation,phosphenes,reconstruction,validation=False):    
#         device = self.prepare(image).device
        
        
#         # Target
#         if self.target == 'image': # Flag for reconstructing input image or target label
#             target = self.prepare(image)
#         # elif self.target == 'label':
#         #     target = labelx
        
#         # Calculate loss
#         loss_stimu = self.stimu_loss(self.prepare(stimulation)) if self.stimu_loss is not None else torch.zeros(1,device=device)
#         # loss_stimu = self.stimu_loss(stimulation.to(device)) if self.stimu_loss is not None else torch.zeros(1,device=device)

#         loss_recon = self.recon_loss(self.prepare(reconstruction),target)
#         # print(reconstruction)
#         # print(target)
#         # print(loss_recon)
#         # exit()

#         loss_total = (1-self.kappa)*loss_recon + self.kappa*loss_stimu
#         return loss_total, torch.mean(loss_recon), loss_stimu