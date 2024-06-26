import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import (
    START_SEQUENCE_TOKEN,
    END_SEQUENCE_TOKEN,
    START_EXAMPLE_TOKEN,
    END_EXAMPLE_TOKEN,
    evaluating_file_paths,
    evaluating_data,
)
from .model import model
from .config import checkpoint_path, device

# import wandb


def eval(checkpoint_path, device, filenames):

    # Load the test data
    # wandb.init(project="hilbert_predictor", job_type="eval")
    test_data = evaluating_data
    test_inputs = [item[0] for item in test_data]
    test_targets = [item[1] for item in test_data]

    test_dataset = TensorDataset(
        torch.tensor(test_inputs, dtype=torch.long),
        torch.tensor(test_targets, dtype=torch.long),
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load the model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    os.makedirs("prediction_plots", exist_ok=True)

    total_correct = 0
    total_tokens = 0
    total_non_zero_correct = 0
    total_non_zero_tokens = 0
    completely_correct = 0
    total_predictions = 0

    with open("predictions.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            [
                "Filename",
                "Input",
                "Predicted",
                "Target",
                "Completely Correct",
                "Non-Zero Accuracy",
            ]
        )

        with torch.no_grad():
            for (src, tgt), filename in zip(test_loader, filenames):
                src, tgt = src.to(device), tgt.to(device)

                # Generate output sequence
                output = model(src)
                _, predicted = torch.max(output.data, -1)

                # Find the end of the actual sequence (ignoring padding)
                tgt_end_idx = (tgt == END_SEQUENCE_TOKEN).nonzero(as_tuple=True)[1][0]
                pred_end_idx = (predicted == END_SEQUENCE_TOKEN).nonzero(as_tuple=True)[
                    1
                ]
                pred_end_idx = (
                    pred_end_idx[0] if pred_end_idx.numel() > 0 else predicted.size(1)
                )

                # Extract the relevant parts of the sequences
                predicted = predicted[:, :pred_end_idx]
                target = tgt[:, :tgt_end_idx]

                # Remove special tokens
                special_tokens = [
                    START_SEQUENCE_TOKEN,
                    END_SEQUENCE_TOKEN,
                    START_EXAMPLE_TOKEN,
                    END_EXAMPLE_TOKEN,
                ]

                def remove_special_tokens(seq):
                    return [token for token in seq if token not in special_tokens]

                predicted_clean = remove_special_tokens(predicted[0].cpu().tolist())
                target_clean = remove_special_tokens(target[0].cpu().tolist())

                # Convert back to tensors
                predicted_clean = torch.tensor(predicted_clean, device=device)
                target_clean = torch.tensor(target_clean, device=device)

                # Truncate to the shorter length
                min_len = min(len(predicted_clean), len(target_clean))
                predicted_clean = predicted_clean[:min_len]
                target_clean = target_clean[:min_len]

                correct = (predicted_clean == target_clean).sum().item()
                total_correct += correct
                total_tokens += len(target_clean)

                non_zero_mask = target_clean != 0
                non_zero_correct = (
                    ((predicted_clean == target_clean) & non_zero_mask).sum().item()
                )
                total_non_zero_correct += non_zero_correct
                total_non_zero_tokens += non_zero_mask.sum().item()

                input_seq = remove_special_tokens(src[0].cpu().tolist())
                predicted_seq = predicted_clean.cpu().numpy()
                target_seq = target_clean.cpu().numpy()

                is_completely_correct = np.array_equal(predicted_seq, target_seq)
                completely_correct += int(is_completely_correct)
                total_predictions += 1

                # Calculate non-zero accuracy for this prediction
                non_zero_mask = target_seq != 0
                non_zero_correct = np.sum((predicted_seq == target_seq) & non_zero_mask)
                non_zero_total = np.sum(non_zero_mask)
                non_zero_accuracy = (
                    non_zero_correct / non_zero_total if non_zero_total > 0 else 0
                )

                csvwriter.writerow(
                    [
                        filename,
                        input_seq,
                        predicted_seq.tolist(),
                        target_seq.tolist(),
                        is_completely_correct,
                        non_zero_accuracy,
                    ]
                )

                print(f"Input: {input_seq}")
                print(f"Predicted: {predicted_seq}")
                print(f"Target: {target_seq}")
                print(f"Completely correct: {is_completely_correct}")
                print(f"Non-zero accuracy: {non_zero_accuracy:.4f}")

    overall_accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    overall_non_zero_accuracy = (
        total_non_zero_correct / total_non_zero_tokens
        if total_non_zero_tokens > 0
        else 0
    )
    completely_correct_percentage = (
        (completely_correct / total_predictions) * 100 if total_predictions > 0 else 0
    )

    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall Non-Zero Accuracy: {overall_non_zero_accuracy:.4f}")
    print(f"Total Correct Predictions: {total_correct} out of {total_tokens}")
    print(
        f"Total Non-Zero Correct Predictions: {total_non_zero_correct} out of {total_non_zero_tokens}"
    )
    print(
        f"Completely Correct Predictions: {completely_correct} out of {total_predictions}"
    )
    print(
        f"Percentage of Completely Correct Predictions: {completely_correct_percentage:.2f}%"
    )

    # wandb.log({
    #     "overall_accuracy": overall_accuracy,
    #     "overall_non_zero_accuracy": overall_non_zero_accuracy,
    #     "completely_correct_percentage": completely_correct_percentage,
    # })

# get the filename from the path without the extension
filenames = [os.path.splitext(os.path.basename(f))[0] for f in evaluating_file_paths]

eval(checkpoint_path, device, filenames)
