import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.vision_transformer import vit_b_16
from Data_Loader_Oxford import data_process
from tqdm import tqdm


def main():
    # Load your dataloaders
    train_loader, val_loader, class_names = data_process()
    num_classes = len(class_names)

    # Load Vision Transformer (ViT-B/16)
    model = vit_b_16(weights="IMAGENET1K_V1")
    model.heads = nn.Linear(model.hidden_dim, num_classes)  # Adjust classifier layer

    # Move the Model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]")

        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=f"{running_loss / len(train_loader):.4f}")

        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]")
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate metrics
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        current_lr = scheduler.get_last_lr()[0]

        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")


if __name__ == "__main__":
    main()
