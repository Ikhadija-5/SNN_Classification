import torch
import os
import lava.lib.dl.slayer as slayer
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import matplotlib.animation as animation
import numpy as np


import numpy as np
import torch

import numpy as np
import torch


import numpy as np
import torch


def compare_ops(net, counts, mse):
    """
    Compare SNN vs ANN, skipping pooling layers, with full table including shapes.
    Uses spike counts collected after testing.
    """

    # 1️⃣ Filter layers (skip pooling)
    layers = [b for b in net.blocks if hasattr(b, 'neuron') and 'pool' not in b.__class__.__name__.lower()]
    counts_used = counts[:len(layers)]

    # 2️⃣ Collect shapes
    layer_shapes = []
    for b in layers:
        if hasattr(b, 'out_shape'):
            layer_shapes.append(b.out_shape)
        elif hasattr(b, 'synapse'):
            if hasattr(b.synapse, 'out_channels'):
                layer_shapes.append((b.synapse.out_channels, 1, 1))
            elif hasattr(b.synapse, 'out_features'):
                layer_shapes.append((b.synapse.out_features, 1, 1))
            else:
                layer_shapes.append((1, 1, 1))
        else:
            layer_shapes.append((1, 1, 1))

    # 3️⃣ Compute synops, activations, MACs
    sdnn_synops = []
    ann_synops = []
    activations = []
    macs = []

    for l, block in enumerate(layers):
        shape = layer_shapes[l]
        prev_count = counts_used[l-1] if l > 0 else counts_used[l]
        activ = np.prod(shape)
        activations.append(activ)

        # Conv layer
        if hasattr(block, 'synapse') and hasattr(block.synapse, 'out_channels'):
            c_out = block.synapse.out_channels
            k = np.prod(block.synapse.kernel_size)
            s = np.prod(block.synapse.stride) if hasattr(block.synapse, 'stride') else 1
            conv_synops = prev_count * c_out * k / s
            sdnn_synops.append(conv_synops)
            ann_synops.append(conv_synops * np.prod(layer_shapes[l-1]) / prev_count if l > 0 else conv_synops)
            macs.append(activ * k)

        # Dense layer
        elif hasattr(block, 'synapse') and hasattr(block.synapse, 'out_features'):
            out_f = block.synapse.out_features
            fc_synops = prev_count * out_f
            sdnn_synops.append(fc_synops)
            ann_synops.append(fc_synops * np.prod(layer_shapes[l-1]) / prev_count if l > 0 else fc_synops)
            macs.append(activ * out_f)

        # Fallback
        else:
            sdnn_synops.append(prev_count * activ)
            ann_synops.append(prev_count * activ)
            macs.append(activ)

    # 4️⃣ Totals
    total_events = np.sum(counts_used)
    total_sdnn = np.sum(sdnn_synops)
    total_ann = np.sum(ann_synops)
    total_activs = np.sum(activations)
    total_macs = np.sum(macs)
    total_neurons = total_activs

    # 5️⃣ Print table
    print(f'|{"-"*77}|')
    print('|', ' '*23,                 '|          SNN           |           ANN           |')
    print(f'|{"-"*77}|')
    print('|', ' '*7, f'|     Shape     |  Events  |    Synops    | Activations|    MACs    |')
    print(f'|{"-"*77}|')
    for i, (c, shape) in enumerate(zip(counts_used, layer_shapes)):
        z, y, x = shape if len(shape) == 3 else (shape[0], 1, 1)
        print(f'| layer-{i} | ({x:3d},{y:3d},{z:3d}) | {c:8.2f} | ', end='')
        if i == 0:
            print(f'{" "*12} | {activations[i]:10.0f} | {" "*10} |')
        else:
            print(f'{sdnn_synops[i]:12.2f} | {activations[i]:10.0f} | {macs[i]:10.0f} |')
    print(f'|{"-"*77}|')
    print(f'|  Total  | {" "*13} | {total_events:8.2f} | {total_sdnn:12.2f} | {total_activs:10.0f} | {total_macs:10.0f} |')
    print(f'|{"-"*77}|')

    # 6️⃣ Summary
    print('\n')
    print(f"MSE              : {mse:.5f} sq. radians")
    print(f"Total neurons    : {total_neurons}")
    print(f"Events sparsity  : {total_activs / total_events:.2f}x" if total_events > 0 else "Events sparsity: N/A")
    print(f"Synops sparsity  : {total_ann / total_sdnn:.2f}x" if total_sdnn > 0 else "Synops sparsity: N/A")




#################Generate Spike predictions animations###############################
def generate_and_save_gifs(net, input, device, trained_folder, num_gifs=5):
    """
    Generates and saves GIFs for the input and output of a neural network and prints their save paths.
    
    Parameters:
    - net: The trained neural network.
    - input: The input tensor to the network.
    - device: The device (CPU/GPU) where the network is loaded.
    - trained_folder: The folder where the trained model and GIFs are saved.
    - num_gifs: The number of GIFs to generate (default is 5).
    """
    
    # Ensure the output directory for GIFs exists
    gif_output_dir = os.path.join(trained_folder, 'gifs_predictions')
    os.makedirs(gif_output_dir, exist_ok=True)

    # Load the trained model
    net.load_state_dict(torch.load(trained_folder + '/network.pt'))
    net.export_hdf5(trained_folder + '/network.net')
    
    # Generate the output from the network
    output = net(input.to(device))

    # Loop to generate and save GIFs for input and output
    for i in range(num_gifs):
        # Convert the input and output tensors to event representations
        inp_event = slayer.io.tensor_to_event(input[i].cpu().data.numpy().reshape(2, 260, 360, -1))
        out_event = slayer.io.tensor_to_event(output[i].cpu().data.numpy().reshape(1, 2, -1))
        
        # Generate animations for the input and output events
        inp_anim = inp_event.anim(plt.figure(figsize=(5, 5)), frame_rate=240)
        out_anim = out_event.anim(plt.figure(figsize=(10, 5)), frame_rate=240)
        
        # Save the generated GIFs
        inp_gif_path = os.path.join(gif_output_dir, f'inp{i}.gif')
        out_gif_path = os.path.join(gif_output_dir, f'out{i}.gif')
        
        inp_anim.save(inp_gif_path, animation.PillowWriter(fps=24), dpi=300)
        out_anim.save(out_gif_path, animation.PillowWriter(fps=24), dpi=300)

        # Print statements indicating successful GIF generation and save location
        print(f"Generated GIF for input {i} and saved at: {inp_gif_path}")
        print(f"Generated GIF for output {i} and saved at: {out_gif_path}")









############################## Save Experiment Details to a Txt File ###############################
def save_experiment_details(output_dir, args,stats_str):
    """
    Save experiment parameters and details to a text file.

    Args:
        output_dir (str): The directory to save the details.
        args (Namespace): The command-line arguments passed to the script.
        stats (slayer.utils.LearningStats): The stats object containing training/testing metrics.
    """
    #stats = slayer.utils.stats.LearningStats()
    #statsl = slayer.utils.stats.LearningStats
    
    
    file_path = os.path.join(output_dir, "experiment_details.txt")
    with open(file_path, "w") as f:
        f.write("Experiment Details\n")
        f.write("===================\n")
        f.write(f"Network: {args.network}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Max TimeStamps: {args.max_timeStamps}\n")
        f.write(f"Sample Length: {args.sample_length}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write("\n")

    

    print(f"Experiment details saved to {file_path}")       