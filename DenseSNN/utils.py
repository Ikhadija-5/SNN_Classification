import torch
import os
import lava.lib.dl.slayer as slayer
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import matplotlib.animation as animation
import numpy as np


##################Generate Input GIFs##################################################
def generate_gif_from_spikes(testing_set, output_dir, frame_rate=240, fps=24, dpi=300):
    """
    Generate GIFs from spike tensors in a testing set.

    Args:
        testing_set (Dataset): The dataset containing spike tensors and labels.
        output_dir (str): Directory to save the generated GIFs.
        frame_rate (int): Frame rate for the animation.
        fps (int): Frames per second for the output GIF.
        dpi (int): Resolution of the output GIF.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Group samples by class
    class_dict = {}
    for idx in range(len(testing_set)):
        spike_tensor, label = testing_set[idx]
        if label not in class_dict:
            class_dict[label] = []
        class_dict[label].append(spike_tensor)

    # Generate GIFs for the first 3 examples of each class
    for label, spike_tensors in class_dict.items():
        for example_idx, spike_tensor in enumerate(spike_tensors[:3]):  # Select first 3 examples
            spike_tensor = spike_tensor.reshape(2, 260, 360, -1)
            event = slayer.io.tensor_to_event(spike_tensor.cpu().data.numpy())

            # Generate animation
            fig = plt.figure(figsize=(5, 5))
            anim = event.anim(fig, frame_rate=frame_rate)

            # Save animation as GIF
            gif_path = os.path.join(output_dir, f'class{label}_example{example_idx}.gif')
            anim.save(gif_path, writer='pillow', fps=fps, dpi=dpi)
            plt.close(fig)  # Close the figure to free memory

            print(f"Saved GIF for class {label}, example {example_idx}: {gif_path}")


############################ Save Experiment Details to a Txt File ###############################
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

        f.write("Final Training Statistics\n")
        f.write("===========================\n")
        f.write("Stats Details: {stats_str}\n")
        # f.write(f"Max Training Accuracy: {stats.max_accuracy:.4f}\n")
        # f.write(f"Max Testing Accuracy: {stats.max_accuracy:.4f}\n")
        # f.write(f"Min Training Loss: {stats.min_loss:.4f}\n")
        # f.write(f"Min Testing Loss: {stats.min_loss:.4f}\n")
        # f.write(f"Stats: {stats}\n")
        # f.write("\n")
        # f.write("Additional Notes:\n")
        #f.write(f"Write any other important details about the experiment here.{statsl.save(path=file_path)}\n")

    print(f"Experiment details saved to {file_path}")
  
    
    
def compare_ops(net, counts, mse):
    shapes = [b.shape for b in net.blocks if hasattr(b, 'neuron')]

    # synops calculation
    snn_synops = []
    ann_synops = []
    for l in range(1, len(net.blocks)):
        if hasattr(net.blocks[l], 'neuron') is False:
            break
        conv_synops = ( # ignoring padding
                counts[l-1]
                * net.blocks[l].synapse.out_channels
                * np.prod(net.blocks[l].synapse.kernel_size)
                / np.prod(net.blocks[l].synapse.stride)
            )
        snn_synops.append(conv_synops)
        ann_synops.append(conv_synops*np.prod(net.blocks[l-1].shape)/counts[l-1])
        # ann_synops.append(conv_synops*np.prod(net.blocks[l-1].shape)/counts[l-1]*np.prod(net.blocks[l].synapse.stride))
        
    for l in range(l+1, len(net.blocks)):
        fc_synops = counts[l-2] * net.blocks[l].synapse.out_channels
        sdnn_synops.append(fc_synops)
        ann_synops.append(fc_synops*np.prod(net.blocks[l-1].shape)/counts[l-2])

    # event and synops comparison
    total_events = np.sum(counts)
    total_synops = np.sum(snn_synops)
    total_ann_activs = np.sum([np.prod(s) for s in shapes])
    total_ann_synops = np.sum(ann_synops)
    total_neurons = np.sum([np.prod(s) for s in shapes])
    steps_per_inference = 1

    print(f'|{"-"*77}|')
    print('|', ' '*23,                 '|          SNN           |           ANN           |')
    print(f'|{"-"*77}|')
    print('|', ' '*7, f'|     Shape     |  Events  |    Synops    | Activations|    MACs    |')
    print(f'|{"-"*77}|')
    for l in range(len(counts)):
        print(f'| layer-{l} | ', end='')
        if len(shapes[l]) == 3: z, y, x = shapes[l]
        elif len(shapes[l]) == 1:
            z = shapes[l][0]
            y = x = 1
        print(f'({x:-3d},{y:-3d},{z:-3d}) | {counts[l]:8.2f} | ', end='')
        if l==0:
            print(f'{" "*12} | {np.prod(shapes[l]):-10.0f} | {" "*10} |')
        else:
            print(f'{sdnn_synops[l-1]:12.2f} | {np.prod(shapes[l]):10.0f} | {ann_synops[l-1]:10.0f} |')
    print(f'|{"-"*77}|')
    print(f'|  Total  | {" "*13} | {total_events:8.2f} | {total_synops:12.2f} | {total_ann_activs:10.0f} | {total_ann_synops:10.0f} |')
    print(f'|{"-"*77}|')

    print('\n')
    print(f'MSE            : {mse:.5} sq. radians')
    print(f'Total neurons  : {total_neurons}')
    print(f'Events sparsity: {total_ann_activs/total_events:5.2f}x')
    print(f'Synops sparsity: {total_ann_synops/total_synops:5.2f}x')