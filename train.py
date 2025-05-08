import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SubjectDataset
from models import Simple1DCNN
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score

import wandb

def train(args):

    #wandb.init(project="BodyInMovement", config=vars(args))
    #wandb.login()
    #config=wandb.config

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #featidx = list(range(0,306)) # use all features  
    #featidx = [i-2 for i in range(2,20)]#[2, 20, 38]] # use a subset of features
    #featidx = [i-2 for i in [29,32,35,83,86,89,313,314,315]] # hip flexor columns
    #featidx = list(range(0,312)) # use all features  
    featidx = [i-2 for i in [29,32,35,83,86,89,307, 308, 309,313,314, 315]] # hip flexor columns
        #featidx = list(range(0,306)) # use all features  
    #featidx = [i-2 for i in range(2,20)]#[2, 20, 38]] # use a subset of features
    # featidx = [i-2 for i in [29,30,31,32,33,34,35,36,37,83,84,85,86,87,88,89,90,91]] # all hip data
    # featidx = [i-2 for i in [218,219,220,221,222,223,224,225,226]] # all head data
    # featidx = [i-2 for i in [227,228,229,230,231,232,233,234,235]] # all neck data
    # featidx = [i-2 for i in [56,57,58,59,60,61,62,63,64,110,111,112,113,114,115,116,117,118]] # all feet data
    # featidx = [i-2 for i in [29,30,31,32,33,34,35,36,37,56,57,58,59,60,61,62,63,64,83,84,85,86,87,88,89,90,91,110,111,112,113,114,115,116,117,118,
    #                         218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235]] # hip, head, neck, feet data
    #featidx = list(range(0,312)) # use all features  



    # Datasets
    excludeIds = ['p021', 'p028', 'p051', 'p288', 'p181',
                  'p160', 'p170', 'p138', 'p036','p122', 'p176', 'p052', 'p137','p048', 'p071', 'p165'] #missing Accel data
    splits = {'train': 80, 'val': 20, 'test': 0}
    seed = 42
    train_dataset = SubjectDataset(args.db_folder, excludeIds, mode='train', 
                 winLength = args.window_length, featuresIdx = featidx,
                 splitPcts= splits, seed=seed)

    val_dataset = SubjectDataset(args.db_folder, excludeIds, mode='val', 
                 winLength = args.window_length, featuresIdx = featidx,
                 splitPcts= splits, seed=seed)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize model
    model = Simple1DCNN(num_sensors=len(featidx), num_classes=3).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    # Load checkpoint if resuming training
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

        # Update the learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate
        print(f"Resuming training from epoch {start_epoch} \
              with learning rate {args.learning_rate}")


    # Training loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    val_f1_scores = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:

            X, Xc, y = batch
            X, Xc, y = X.to(device), Xc.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X, Xc)
            loss = criterion(outputs, y + 1)  # Shift labels by +1 for CrossEntropyLoss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y.size(0)
            train_correct += (predicted == (y + 1)).sum().item()

            # Update progress bar
            pbar.set_postfix({'loss': f"{train_loss/train_total:.4f}", 
                          'acc': f"{train_correct/train_total:.4f}"})

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        wandb.log({
            "epoch":epoch+1,
            "train_loss":train_loss,
            "train_accuracy":train_accuracy
        })

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_targets = [],[]

        with torch.no_grad():
            for batch in val_loader:
                X, Xc, y = batch
                X, Xc, y = X.to(device), Xc.to(device), y.to(device)

                outputs = model(X, Xc)
                loss = criterion(outputs, y + 1)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += (predicted == (y + 1)).sum().item()

                # Accumulate predictions and targets for F1 score calculation
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend((y + 1).cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        val_f1 = f1_score(all_targets, all_preds, average='macro')

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)

        wandb.log({
            "val_loss":val_loss,
            "val_accuracy":val_accuracy,
            "val_f1_score":val_f1

        })

        print(f"Epoch [{epoch+1}/{args.epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')

    
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

   # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and validation accuracy')

    plt.tight_layout()
    plt.savefig('training_plot.png')
    wandb.log({"training_plot": wandb.Image("training_plot.png")})
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the baseline model")

    parser.add_argument('--db_folder', type=str, default='/home/ibroto/Documents/SMC/SPIS/BodyInTransit/DATASET/train', help='the path to the dataset')
    parser.add_argument('--window_length', type=int, default=125, help='the size of the windows to extract as inputs for the models')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every n epochs')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')

    args = parser.parse_args()
    train(args)
