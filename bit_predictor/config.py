# Model initialization
from pathlib import Path
import torch


batch_size = 2
if torch.cuda.is_available():
    batch_size = 2048
d_model = 32
nhead = 4
num_layers = 6
dim_feedforward = 128
max_seq_length = 72
max_context_length = 64
max_prediction_length = 8
dropout_rate = 0.12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = Path("bit_checkpoint.pt")
