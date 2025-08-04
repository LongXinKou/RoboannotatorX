import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding"""

    def __init__(self, hidden_size, max_seq_length=5000):
        super().__init__()

        # Create the positional encoding matrix
        pe = torch.zeros(max_seq_length, hidden_size)
        position = torch.arange(0, max_seq_length).unsqueeze(1)  # [max_seq_len, 1]
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))

        # Apply sine to even indices; cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: [max_seq_length, 1, hidden_size]
        pe = pe.unsqueeze(1)

        # Register as buffer (not a model parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [seq_len, batch_size, hidden_size]
        Returns:
            x with positional encoding added
        """
        return x + self.pe[:x.size(0)]


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()

        self.hidden_size = hidden_size

        # Positional encoding layer
        self.pos_encoder = PositionalEncoding(hidden_size)
        self.dropout = nn.Dropout(p=dropout)

        # Define a single Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            batch_first=False  # Input shape: [seq_len, batch, hidden]
        )

        # Stack multiple Transformer Encoder Layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x, src_mask=None, src_padding_mask=None):
        """
        Args:
            x: Input sequence of shape [seq_len, batch_size, hidden_size]
            src_mask: Attention mask for the input
            src_padding_mask: Mask for padded positions
        Returns:
            Encoded output of shape [seq_len, batch_size, hidden_size]
        """
        x = self.pos_encoder(x)  # Add positional encoding
        x = self.dropout(x)      # Apply dropout
        output = self.transformer_encoder(x, mask=src_mask,
                                          src_key_padding_mask=src_padding_mask)
        return output


def get_transformer_state(model_params, keys_to_match=None):
    """Extract parameters related to the Transformer encoder"""
    if keys_to_match is None:
        keys_to_match = [
            'transformer_encoder',  # Transformer encoder layers
            'pos_encoder',          # Positional encoding buffer
        ]

    transformer_state_dict = {}

    # Iterate through model named parameters and collect matching ones
    for name, param in model_params:
        if any(key in name for key in keys_to_match):
            transformer_state_dict[name] = param

    return transformer_state_dict