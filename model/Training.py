import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter


def training(model, superpixel_vectors, epochs, learning):
    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning)

    # Initialize TensorBoard writer
    writer = SummaryWriter()

    # Training loop
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = superpixel_vectors.size(0)  # Total samples based on superpixel vectors

        epoch_start_time = time.time()

        # Progress bar for one epoch
        with tqdm(total=total_samples, desc=f"Epoch {epoch + 1}/{epochs}", unit='batch') as pbar:
            for i in range(total_samples):
                # Use the superpixel vector for training
                inputs = superpixel_vectors[i].unsqueeze(0).to(device)  # Add batch dimension

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels[i].to(device))  # Ensure you have corresponding labels

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update statistics
                running_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=running_loss / (i + 1))

        epoch_end_time = time.time()
        epoch_loss = running_loss / total_samples
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')

    total_duration = time.time() - start_time
    print(f"Training completed in {total_duration:.2f} seconds")

    # Close TensorBoard writer
    writer.close()

    return model



