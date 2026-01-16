import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os 
import sys
from qat_model import MNISTNet

def load_data(data_dir):
    print(f"Loading data from {data_dir}...")

    #reads .npy files into numpy arrays
    train_images = np.load(f'{data_dir}/mnist_14x14_train.npy')  # (60000, 14, 14)
    train_labels = np.load(f'{data_dir}/train_labels.npy')       # (60000,)
    test_images = np.load(f'{data_dir}/mnist_14x14_test.npy')    # (10000, 14, 14)
    test_labels = np.load(f'{data_dir}/test_labels.npy')         # (10000,)

    #convert numpy arrays to tensors
    train_images_tensor = torch.tensor(train_images, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    test_images_tensor = torch.tensor(test_images, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)


    #dataset[i] = (image_i, label_i)
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, test_loader

#train model for one epoch
def train_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_i, (images, labels) in enumerate(train_loader):
        
        #reset gradient for each batch
        optimizer.zero_grad()

        #forward pass
        outputs = model(images)

        #compute loss
        loss = criterion(outputs, labels)

        #backward pass
        loss.backward()

        #update weights
        optimizer.step()

        total_loss += loss.item()
        value, predicted = outputs.max(dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_i % 200 == 0:
            print(f'  Batch {batch_i}/{len(train_loader)}, Loss: {loss.item():.4f}')
    avg_loss = total_loss/len(train_loader)
    accuracy = 100.0 * correct / total
    print(f'Epoch {epoch}: Train Loss = {avg_loss:.4f}, Train Accuracy = {accuracy:.2f}%')
    return avg_loss, accuracy


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            value, predicted = outputs.max(dim=1)
            
            # Count correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100.0 * correct / total
    return accuracy

def main():
    DATA_DIR = '../data'
    WEIGHTS_DIR = 'weights'
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.005
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    #load data
    train_loader, test_loader = load_data(DATA_DIR)

    #create model
    print("\nCreating model...")
    model = MNISTNet()
    print(model)

    #loss function
    criterion = nn.CrossEntropyLoss()

    #optimizer (Adaptive Moment Estimation)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    print("\nStarting training...")
    best_accuracy = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, epoch)
        test_accuracy = evaluate(model, test_loader)
        print(f'Epoch {epoch}: Test Accuracy = {test_accuracy:.2f}%\n')

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), f'{WEIGHTS_DIR}/model_best.pth')
            print(f'New best model saved! (accuracy: {best_accuracy:.2f}%)')
    torch.save(model.state_dict(), f'{WEIGHTS_DIR}/model_final.pth')
    print("\n" + "="*50)
    print(f"Training complete!")
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    print(f"Model saved to: {WEIGHTS_DIR}/model_best.pth")
    print("="*50)

if __name__ == "__main__":
    main()