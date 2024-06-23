import torch
import torch.nn as nn
from .config import max_seq_length, d_model


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.encoding = nn.Parameter(
            torch.zeros(max_seq_length, 2), requires_grad=False
        )

        # Initialize the encoding with normalized pixel positions
        positions = torch.arange(max_seq_length).unsqueeze(1)
        self.encoding[:, 0] = (
            positions % 4
        ).squeeze() / 3 * 2 - 1  # Normalize X position to [-1, 1]
        self.encoding[:, 1] = torch.where(
            positions.squeeze() < 4,
            (positions // 4).squeeze() / 3 * 2
            - 1,  # Normalize Y position to [-1, 1] for the first row
            torch.zeros(max_seq_length),  # Set Y position to 0 for the rest
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        return self.encoding[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)


def test_positional_encoding():
    # Create a sample input tensor
    batch_size = 2
    seq_length = 10
    input_tensor = torch.randn(batch_size, seq_length, d_model)

    # Create an instance of PositionalEncoding
    pos_encoding = PositionalEncoding(d_model, max_seq_length)

    # Get the position embeddings for the input tensor
    pos_embeddings = pos_encoding(input_tensor)

    # Check the shape of the position embeddings
    assert pos_embeddings.shape == (
        batch_size,
        seq_length,
        2,
    ), f"Expected shape (batch_size, seq_length, 2), but got {pos_embeddings.shape}"

    # Check the values of the position embeddings
    expected_x = torch.tensor(
        [
            [
                -1.0000,
                -0.3333,
                0.3333,
                1.0000,
                -1.0000,
                -0.3333,
                0.3333,
                1.0000,
                -1.0000,
                -0.3333,
            ]
        ]
    )
    expected_y = torch.tensor(
        [
            [
                -1.0000,
                -1.0000,
                -1.0000,
                -1.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ]
        ]
    )

    assert torch.allclose(
        pos_embeddings[0, :, 0], expected_x, atol=1e-4
    ), f"Expected x-values: {expected_x}, but got {pos_embeddings[0, :, 0]}"
    assert torch.allclose(
        pos_embeddings[0, :, 1], expected_y, atol=1e-4
    ), f"Expected y-values: {expected_y}, but got {pos_embeddings[0, :, 1]}"

    print("Positional encoding test passed!")


if __name__ == "__main__":
    test_positional_encoding()
