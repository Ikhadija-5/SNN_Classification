import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import lava.lib.dl.slayer as slayer
import time

#from dataset loader
from dataset import augment, EyeDataset
from network import Conv_SNN, Dense_SNN, VGG_SNN
from utils import save_experiment_details
from utils import compare_ops


torch.cuda.set_device(0)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train an SNN with dynamic input handling")
    parser.add_argument('--network', type=str, required=True, choices=['dense', 'conv','vgg'], help="Choose network type")
    parser.add_argument('--trained_folder', type=str, default='Trained', help="Output folder for trained models")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--max_timeStamps', type=int, default=100, help="Maximum time stamps for events")
    parser.add_argument('--sample_length', type=int, default=100, help="Sample length in ms")
    #parser.add_argument('--gpu', type=int, default=-1, choices=[-1, 0, 1], help="GPU device index (0 or 1), -1 for CPU")

    args = parser.parse_args()


    trained_folder = args.trained_folder
    gif_output_dir = os.path.join(trained_folder, 'gifs')
    os.makedirs(gif_output_dir, exist_ok=True)

 
    
    training_set = EyeDataset(
        train=True, sampling_time=1, sample_length=args.sample_length, transform=augment, max_timeStamps=args.max_timeStamps
    )
    testing_set = EyeDataset(
        train=False, sampling_time=1, sample_length=args.sample_length, max_timeStamps=args.max_timeStamps
    )


    train_loader = DataLoader(dataset=training_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testing_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda:0')

    # Initialize appropriate network
    if args.network == 'dense':
        net = Dense_SNN()
    elif args.network == 'conv':
        net = Conv_SNN()
    elif args.network == 'vgg':
        net = VGG_SNN()
    


    net.to(device)

    # --- Compute parameters ---
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")



    # Define optimizer, loss, and stats
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    stats = slayer.utils.LearningStats()

    # Initialize Assistant
    error = slayer.loss.SpikeRate(
        true_rate=0.2, false_rate=0.03, reduction='sum'
    ).to(device)

    assistant = slayer.utils.Assistant(
        net, error, optimizer, stats,
        classifier=slayer.classifier.Rate.predict, count_log=True
    )

    



# Training loop
for epoch in range(args.epochs):
    epoch_start = time.time()  # Start timing this epoch

    for i, (input, label) in enumerate(train_loader):  # Training loop
        input = input.to(device)
        label = label.to(device)
        output, count = assistant.train(input, label)
        header = [
            'Event rate : ' +
            ', '.join([f'{c.item():.4f}' for c in count.flatten()])
        ]
        stats.print(epoch, iter=i, header=header, dataloader=train_loader)



    for i, (input, label) in enumerate(test_loader):  # Testing loop
        output, count = assistant.test(input, label)
        header = [
            'Event rate : ' +
            ', '.join([f'{c.item():.4f}' for c in count.flatten()])
        ]
        stats.print(epoch, iter=i, header=header, dataloader=test_loader)
        # print(stats)

    # Calculate epoch duration
    epoch_end = time.time()
    epoch_duration = epoch_end - epoch_start
    print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_duration:.2f} seconds")

    # Save the model if best accuracy improves
    if stats.testing.best_accuracy:
        model_path = os.path.join(trained_folder, 'network.pt')
        torch.save(net.state_dict(), model_path)
        print(f"Model saved to {model_path} with best accuracy: {stats.testing.best_accuracy:.4f}")

    # Save stats and plots
    stats_str = str(stats).replace("| ", "\n")

    stats.update()
    stats.save(trained_folder + '/')
    stats.plot(path=trained_folder + '/')
    net.grad_flow(trained_folder + '/')
    




counts = []
for i, (input, label) in enumerate(test_loader):
    _, count = assistant.test(input, label)
    count = (count.flatten() / (input.shape[-1]-1) / input.shape[0]).tolist()
    counts.append(count)

counts = np.mean(counts, axis=0)
print(counts)
compare_ops(net, counts, mse=stats.testing.min_loss)



#python train.py --network conv --trained_folder Trained_60  --epochs  100 --batch_size 2 --max_timeStamps 60 --sample_length 6