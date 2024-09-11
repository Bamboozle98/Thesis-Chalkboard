import torch


def evaluation(model, val_loader, class_names):

    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Get the predicted class by finding the max logit
            _, predicted = torch.max(outputs, 1)

            # Print actual and predicted labels for each batch
            for i in range(len(labels)):
                actual_label = class_names[labels[i].item()]
                predicted_label = class_names[predicted[i].item()]
                print(f"Actual label: {actual_label}, Predicted label: {predicted_label}")

            # Update total and correct predictions
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples
    print(f'Accuracy on the validation set: {accuracy * 100:.2f}%')
