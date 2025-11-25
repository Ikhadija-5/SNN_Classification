import torch
# import slayer from lava-dl
import lava.lib.dl.slayer as slayer
import matplotlib.pyplot as plt
import os
import glob
import zipfile
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



class Conv_SNN(torch.nn.Module):
    def __init__(self):
        super(Conv_SNN, self).__init__()

        neuron_params = {
                'threshold'     : 1.25,
                'current_decay' : 0.25,
                'voltage_decay' : 0.03,
                'tau_grad'      : 0.03,
                'scale_grad'    : 3,
                'requires_grad' : True,     
            }
        neuron_params_drop = {**neuron_params, 'dropout' : slayer.neuron.Dropout(p=0.05),}
        


        self.blocks = torch.nn.ModuleList([
            # First convolution block
            slayer.block.cuba.Conv(neuron_params, 2, 8, kernel_size=5, stride=1, padding=0),
            slayer.block.cuba.Pool(neuron_params, 2),  # Pooling after first convolution
            
            # Second convolution block
            slayer.block.cuba.Conv(neuron_params, 8, 8, kernel_size=5, stride=1, padding=0),

            slayer.block.cuba.Conv(neuron_params, 8, 2, kernel_size=5, stride=1, padding=0),
            slayer.block.cuba.Pool(neuron_params, 2),  # Pooling after second convolution
            
            # Flatten operation (manual insertion during forward pass)
            slayer.block.cuba.Flatten(),
            
            # First dense (fully connected) block
            slayer.block.cuba.Dense(neuron_params_drop, 10200, 512, weight_norm=True, delay=True),
            
            # Second dense block
            slayer.block.cuba.Dense(neuron_params_drop, 512, 512, weight_norm=True, delay=True),
            
            # Output layer
            slayer.block.cuba.Dense(neuron_params, 512, 2, weight_norm=True)
            ])
    
    def forward(self, spike):
        count = []
        for block in self.blocks:
            spike = block(spike)
            # print("Layer-wise shape",spike.shape)
            count.append(torch.sum(spike > 0).item())
            # count.append(torch.mean(spike).item())
        # return spike
        return spike, torch.FloatTensor(count).reshape((1, -1)).to(spike.device)
    
    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))   

