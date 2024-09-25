import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from FeatureVectors import assign_features_to_superpixels  # Import the function


def training(vit_model, cnn_model, train_loader, epochs, learning):
    # Move the models to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    vit_model = vit_model.to(device)
    cnn_model = cnn_model.to(device)

    # Set CNN model to evaluation mode as we will not be training it
    cnn_model.eval()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vit_model.parameters(), lr=learning)

    # Initialize TensorBoard writer
    writer = SummaryWriter()

    # Training loop
    num_epochs = epochs
    start_time = time.time()

    for epoch in range(num_epochs):
        vit_model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        epoch_start_time = time.time()

        # Progress bar for one epoch
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch') as pbar:
            for superpixel_map, image, labels in train_loader:
                image, labels = image.to(device), labels.to(device)

                # Forward pass through the SuperpixelCNN
                with torch.no_grad():  # CNN is not being trained, so no need to compute gradients
                    cnn_features = cnn_model(image)  # Get feature vectors from CNN

                # Assign CNN features to superpixels based on the superpixel_map
                superpixel_vectors = assign_features_to_superpixels(cnn_features, superpixel_map)

                # Pass the superpixel feature vectors to the Vision Transformer
                outputs = vit_model(superpixel_vectors)

                # Compute the loss
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate statistics for accuracy and loss
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
                    writer.add_scalar('Accuracy/train', correct_predictions / total_samples * 100,
                                      epoch * len(train_loader) + pbar.n)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples
        print(
            f'\nEpoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}%, Duration: {epoch_duration:.2f} seconds')

    total_duration = time.time() - start_time
    print(f"Training completed in {total_duration:.2f} seconds")

    # Close TensorBoard writer
    writer.close()

    return vit_model


