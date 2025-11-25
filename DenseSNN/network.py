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


    
class Dense_SNN(torch.nn.Module):
    def __init__(self):
        super(Dense_SNN, self).__init__()

        neuron_params = {
                'threshold'     : 1.25,
                'current_decay' : 0.25,
                'voltage_decay' : 0.03,
                'tau_grad'      : 0.03,
                'scale_grad'    : 3,
                'requires_grad' : False,
            }
        # neuron_params_drop = {
        #         **neuron_params,
        #         'dropout' : slayer.neuron.Dropout(p=0.05),
        #     }
        neuron_params_drop = {**neuron_params}

        self.blocks = torch.nn.ModuleList([
                slayer.block.cuba.Dense(
                    neuron_params_drop, 260*360*2, 512,
                    weight_norm=True, delay=True
                ),
                slayer.block.cuba.Dense(
                    neuron_params_drop, 512, 512,
                    weight_norm=True, delay=True
                ),
                slayer.block.cuba.Dense(
                    neuron_params, 512, 2,
                    weight_norm=True
                ),
            ])

    def forward(self, spike):
        count = []
        for block in self.blocks:
            spike = block(spike)
            count.append(torch.mean(spike).item())
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
            
        
   