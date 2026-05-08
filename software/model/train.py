import argparse
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from qat_model import MNISTNet


def load_data(data_dir, batch_size):
    print(f"Loading data from {data_dir}...")

    train_images = np.load(f"{data_dir}/mnist_14x14_train.npy")
    train_labels = np.load(f"{data_dir}/train_labels.npy")
    test_images = np.load(f"{data_dir}/mnist_14x14_test.npy")
    test_labels = np.load(f"{data_dir}/test_labels.npy")

    train_images_tensor = torch.tensor(train_images, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    test_images_tensor = torch.tensor(test_images, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_i % 200 == 0:
            print(f"  Batch {batch_i}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Train Accuracy = {accuracy:.2f}%")
    return avg_loss, accuracy


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = outputs.max(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100.0 * correct / total
    return accuracy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train MNIST QAT model for uTPU")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-out", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    weights_dir = os.path.join(script_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    train_loader, test_loader = load_data(data_dir, args.batch_size)

    print("\nCreating model...")
    model = MNISTNet()
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("\nStarting training...")
    best_accuracy = 0.0

    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, train_loader, criterion, optimizer, epoch)
        test_accuracy = evaluate(model, test_loader)
        print(f"Epoch {epoch}: Test Accuracy = {test_accuracy:.2f}%\n")

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), f"{weights_dir}/model_best.pth")
            print(f"New best model saved! (accuracy: {best_accuracy:.2f}%)")

    torch.save(model.state_dict(), f"{weights_dir}/model_final.pth")
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    print(f"Model saved to: {weights_dir}/model_best.pth")
    print("=" * 50)

    if args.metrics_out:
        metrics = {
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "best_accuracy_pct": round(best_accuracy, 4),
            "best_epoch": best_epoch,
        }
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved training metrics: {args.metrics_out}")


if __name__ == "__main__":
    main()
