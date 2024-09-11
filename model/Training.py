import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter

def training(model, train_loader, epochs, learning):
    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning)

    # Initialize TensorBoard writer for useful UI stuff during training. (Progress bar, time in seconds, etc...)
    writer = SummaryWriter()

    # Training loop
    num_epochs = epochs  # Epochs are adjusted in the main.py
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        epoch_start_time = time.time()

        # Progress bar for one epoch
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch') as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate statistical results
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                running_loss += loss.item()

                # Update progress bar and statistics
                pbar.update(1)
                pbar.set_postfix(loss=running_loss / (pbar.n + 1), accuracy=correct_predictions / total_samples * 100)

                # Log to TensorBoard
                if pbar.n % 10 == 0:  # Log every 10 batches
                    writer.add_scalar('Loss/train', running_loss / (pbar.n + 1), epoch * len(train_loader) + pbar.n)
                    writer.add_scalar('Accuracy/train', correct_predictions / total_samples * 100, epoch * len(train_loader) + pbar.n)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples
        print(f'\nEpoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%, Duration: {epoch_duration:.2f} seconds')

    total_duration = time.time() - start_time
    print(f"Training completed in {total_duration:.2f} seconds")

    # Close TensorBoard writer
    writer.close()

    return model

