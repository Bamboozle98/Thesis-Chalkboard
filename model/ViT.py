import torch
import torch.nn as nn


class CustomTransformerModel(nn.Module):
    def __init__(self, num_classes, num_superpixel_features, num_layers=6, nhead=8):
        """
        Custom Transformer model for superpixel feature classification.

        Parameters:
        - num_classes (int): Number of classes in the dataset.
        - num_superpixel_features (int): Number of features per superpixel.
        - num_layers (int): Number of transformer encoder layers.
        - nhead (int): Number of attention heads.
        """
        super(CustomTransformerModel, self).__init__()

        # Define a transformer encoder
        self.transformer = nn.Transformer(
            d_model=num_superpixel_features,
            nhead=nhead,
            num_encoder_layers=num_layers
        )

        # Define a classification head (simple fully connected layer)
        self.fc = nn.Linear(num_superpixel_features, num_classes)

        # Optional: Positional encoding (useful for transformers dealing with sequences)
        self.positional_encoding = nn.Parameter(torch.zeros(1000, num_superpixel_features))

    def forward(self, src):
        """
        Forward pass for the transformer model.

        Parameters:
        - src (Tensor): Input tensor of shape (sequence_length, batch_size, num_superpixel_features)

        Returns:
        - Output class predictions
        """
        # Add positional encoding (optional)
        src = src + self.positional_encoding[:src.size(0), :]

        # Pass the input through the transformer encoder
        transformer_output = self.transformer(src)

        # Take the mean of the transformer output across sequence length (global average pooling)
        pooled_output = transformer_output.mean(dim=0)

        # Pass the pooled output through the classification head
        output = self.fc(pooled_output)

        return output


# Example usage
def transformer_model(class_names, num_superpixel_features):
    """
    Load a custom Transformer model for classifying superpixel feature vectors.

    Parameters:
    - class_names (list): List of class names in your dataset.
    - num_superpixel_features (int): The number of features per superpixel.

    Returns:
    - model: The custom Transformer model.
    """
    num_classes = len(class_names)  # Number of classes in your dataset
    model = CustomTransformerModel(num_classes=num_classes, num_superpixel_features=num_superpixel_features)

    return model


