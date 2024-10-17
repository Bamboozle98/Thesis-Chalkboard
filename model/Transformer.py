import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, d_model=512, nhead=8, num_encoder_layers=3,
                 dropout=0.1):
        super(TransformerEncoder, self).__init__()
        # Ensure d_model is divisible by nhead
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Project in_channels to d_model
        self.input_projection = nn.Linear(in_channels, d_model)  # Adding positions to encodings

        # Learnable Embedding Vector
        self.learned_token = nn.Parameter(torch.rand(1, d_model))

        # Projection to output
        self.output_projection = nn.Linear(d_model, out_channels)

    def forward(self, x):
        x = self.input_projection(x)  # Project input to embedding dimension

        # Append learned token to input
        learned_token = self.learned_token.expand(x.size(0), -1, -1)
        x = torch.cat((learned_token, x), dim=1)

        x = self.transformer_encoder(x)
        # x = self.dropout(x)  # Apply dropout
        # x = x.mean(dim=1)  # Global average pooling

        x = self.output_projection(x[:, 0, :])  # Project to output dimension

        return x



