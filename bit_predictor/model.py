import torch
import torch.nn as nn
from pathlib import Path
from xformers.components import MultiHeadDispatch
from xformers.components.attention import ScaledDotProduct
from .data import NUM_TOKENS, PAD_TOKEN
from .encoding import PositionalEncoding
from torch.utils.checkpoint import checkpoint

from .config import (
    batch_size,
    d_model,
    nhead,
    num_layers,
    dim_feedforward,
    max_seq_length,
    max_context_length,
    max_prediction_length,
    dropout_rate,
    device,
    checkpoint_path
)


class Transformer(nn.Module):
    def __init__(
        self,
        num_tokens,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        max_seq_length,
        dropout_rate,
        device,
    ):
        super().__init__()
        self.device = device
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.nhead = nhead
        self.embedding = nn.Embedding(num_tokens + 1, d_model, padding_idx=PAD_TOKEN)
        self.token_embedding = nn.Embedding(num_tokens, d_model, padding_idx=PAD_TOKEN)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        self.layers = nn.ModuleList(
            [
                self.create_decoder_layer(d_model, nhead, dim_feedforward, dropout_rate)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(d_model, num_tokens + 1)
        self.to(device)

    def create_decoder_layer(self, d_model, nhead, dim_feedforward, dropout_rate):
        attention = ScaledDotProduct(dropout=dropout_rate, causal=True)

        return nn.ModuleDict(
            {
                "self_attn": MultiHeadDispatch(
                    dim_model=d_model,
                    num_heads=nhead,
                    attention=attention,
                    bias=True,
                    residual_dropout=dropout_rate,
                ),
                "ff": nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(dim_feedforward, d_model),
                    nn.Dropout(dropout_rate),
                ),
                "norm1": nn.LayerNorm(d_model),
                "norm2": nn.LayerNorm(d_model),
            }
        )

    def forward(self, src):
        assert src.dim() == 2, f"Expected input to be 2D, but got {src.dim()}D"

        x = self.token_embedding(src)
        assert x.shape == (
            src.shape[0],
            src.shape[1],
            self.d_model,
        ), f"Expected shape {(src.shape[0], src.shape[1], self.d_model)}, but got {x.shape}"

        pos_encoding = self.pos_encoding(x)
        x = x + pos_encoding.repeat(1, 1, self.d_model // 2)  # Repeat the positional encoding along the feature dimension

        batch_size, seq_len, d_model = x.shape

        for i, layer in enumerate(self.layers):
            x = layer["self_attn"](
                x,
                x,
                x,
            )

            x = layer["norm1"](x)

            ff_output = checkpoint(layer["ff"], x)
            x = layer["norm2"](x + ff_output)

        output = self.fc_out(x)
        assert output.shape == (
            batch_size,
            seq_len,
            NUM_TOKENS + 1,
        ), f"Expected shape {(batch_size, seq_len, NUM_TOKENS + 1)}, but got {output.shape}"

        return output


model = Transformer(
    NUM_TOKENS,
    d_model,
    nhead,
    num_layers,
    dim_feedforward,
    max_seq_length,
    dropout_rate,
    device,
)

def test_model_with_zeros():
    # Create a model instance
    model = Transformer(
        NUM_TOKENS,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        max_seq_length,
        dropout_rate,
        device,
    )

    # Create dummy input data (zeros)
    dummy_input = torch.zeros((batch_size, max_seq_length), dtype=torch.long).to(device)

    # Set the model to training mode
    model.train()

    # Create a dummy target (zeros)
    dummy_target = torch.zeros((batch_size, max_seq_length), dtype=torch.long).to(device)

    # Create an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Perform a forward pass
    output = model(dummy_input)

    # Calculate loss
    loss = criterion(output.view(-1, NUM_TOKENS + 1), dummy_target.view(-1))

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Test completed. Loss: {loss.item()}")

if __name__ == "__main__":
    test_model_with_zeros()