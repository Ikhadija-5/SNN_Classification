import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import lava.lib.dl.slayer as slayer
import time  
from sklearn.metrics import precision_score, recall_score

#from your_dataset_loader import YourDataset  # Replace with your dataset loader
from dataset import augment, EyeDataset
from network import Conv_SNN, Dense_SNN
from utils import generate_gif_from_spikes, save_experiment_details





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train an SNN with dynamic input handling")
    parser.add_argument('--network', type=str, required=True, choices=['dense', 'conv'], help="Choose network type")
    parser.add_argument('--trained_folder', type=str, default='Trained', help="Output folder for trained models")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--max_timestamps', type=int, default=100, help="Maximum time stamps for events")
    parser.add_argument('--sample_length', type=int, default=100, help="Sample length in ms")
    args = parser.parse_args()

    # Prepare the training folder and gifs subfolder
    trained_folder = args.trained_folder
    gif_output_dir = os.path.join(trained_folder, 'gifs')
    os.makedirs(gif_output_dir, exist_ok=True)

    # Initialize datasets
    training_set = EyeDataset(
        train=True, sampling_time=1, sample_length=args.sample_length, transform=augment
    )
    testing_set = EyeDataset(
        train=False, sampling_time=1, sample_length=args.sample_length
    )

    train_loader = DataLoader(dataset=training_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testing_set, batch_size=args.batch_size, shuffle=False)

    # Initialize appropriate network
    if args.network == 'dense':
        net = Dense_SNN()
    elif args.network == 'conv':
        net = Conv_SNN()
    net.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Define optimizer, loss, and stats
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    stats = slayer.utils.LearningStats()

    # Initialize Assistant
    error = slayer.loss.SpikeRate(
        true_rate=0.2, false_rate=0.03, reduction='sum'
    ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    assistant = slayer.utils.Assistant(
        net, error, optimizer, stats,
        classifier=slayer.classifier.Rate.predict, count_log=True
    )



# # # Start tracking total training time
# start_time = time.time()
# print("Generating GIFs...")
# generate_gif_from_spikes(testing_set, gif_output_dir)


from sklearn.metrics import precision_score

all_preds = []
all_targets = []


# Training loop
for epoch in range(args.epochs):
    epoch_start = time.time()  # Start timing this epoch

    for i, (input, label) in enumerate(train_loader):  # Training loop
        # print(label.shape)
        output, count = assistant.train(input, label)
        header = [
            'Event rate : ' +
            ', '.join([f'{c.item():.4f}' for c in count.flatten()])
        ]
        stats.print(epoch, iter=i, header=header, dataloader=train_loader)

        
        

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    for i, (input, label) in enumerate(test_loader):
        output, count = assistant.test(input, label)

        # Run classifier
        preds = assistant.classifier(output)

        # Ensure label is on same device
        label = label.to(preds.device)

        # Save for precision/recall later
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(label.detach().cpu().numpy())

        # Accuracy
        correct += (preds == label).sum().item()
        total += label.size(0)

    # Save the model if best accuracy improves
    if stats.testing.best_accuracy:
        model_path = os.path.join(trained_folder, 'network.pt')
        # torch.save(net.state_dict(), model_path)
        print(f"Model saved to {model_path} with best accuracy: {stats.testing.best_accuracy:.4f}")

    # Save stats and plots
    stats_str = str(stats).replace("| ", "\n")

    # stats.update()
    stats.save(trained_folder + '/')
    stats.plot(path=trained_folder + '/')
    net.grad_flow(trained_folder + '/')
    








###########################    Event and Synops comparion with ANN       #############################################
# counts = []
# for i, (input, label) in enumerate(test_loader):
#     _, count = assistant.test(input, label)
#     count = (count.flatten()/(input.shape[-1]-1)/input.shape[0]).tolist() # count skips first events
#     counts.append(count) 
#     print('\rEvent count : ' + ', '.join([f'{c:.4f}' for c in count]), f'| {stats.testing}', end='') 
        
# counts = np.mean(counts, axis=0)

# print(compare_ops(net, counts, mse=stats.testing.min_loss))

####################################################################################################################
