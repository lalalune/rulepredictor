import numpy as np
import os
import json
import pickle

# Token definitions
START_EXAMPLE_TOKEN = 2
END_EXAMPLE_TOKEN = 3
START_INPUT_MATRIX_TOKEN = 4
END_INPUT_MATRIX_TOKEN = 5
START_OUTPUT_MATRIX_TOKEN = 6
END_OUTPUT_MATRIX_TOKEN = 7
START_SEQUENCE_TOKEN = 8
END_SEQUENCE_TOKEN = 9
PAD_TOKEN = 10
NUM_TOKENS = 11

MAX_CONTEXT_LENGTH = 64
MAX_PREDICTION_LENGTH = 8

evaluating_data = None

def pad_sequence(sequence, max_length, pad_value, left_pad=False):
    if left_pad:
        return np.pad(sequence, (max(0, max_length - len(sequence)), 0), 
                      mode='constant', constant_values=pad_value)
    else:
        return np.pad(sequence, (0, max(0, max_length - len(sequence))), 
                      mode='constant', constant_values=pad_value)

def load_and_process_training_data(file_paths):
    processed_data = []

    for file_path in file_paths:
        with open(file_path, "r") as f:
            data = json.load(f)

        train_examples = data["train"]
        test_examples = data["test"]  # Note: test_examples is a list

        for test_example in test_examples:
            # Create a sequence for each test example
            context = [START_SEQUENCE_TOKEN]

            # Add all training examples to the context
            for train_example in train_examples:
                context = context + [
                    START_EXAMPLE_TOKEN,
                    START_INPUT_MATRIX_TOKEN,
                    *train_example['input'],
                    END_INPUT_MATRIX_TOKEN,
                    START_OUTPUT_MATRIX_TOKEN,
                    *train_example['output'],
                    END_OUTPUT_MATRIX_TOKEN,
                    END_EXAMPLE_TOKEN
                ]

            # Left-pad or truncate the context (excluding the test input)
            context = pad_sequence(context, MAX_CONTEXT_LENGTH - 5, PAD_TOKEN, left_pad=True)
            # Add the test input
            context = context.tolist() + [
                START_EXAMPLE_TOKEN,
                START_INPUT_MATRIX_TOKEN,
                *test_example['input'],
                END_INPUT_MATRIX_TOKEN,
                START_OUTPUT_MATRIX_TOKEN
            ]

            # Create target sequence
            target = (
                test_example['output']
                + [END_OUTPUT_MATRIX_TOKEN, END_EXAMPLE_TOKEN, END_SEQUENCE_TOKEN]
            )

            # Right-pad or truncate the target
            target = pad_sequence(target, MAX_PREDICTION_LENGTH, PAD_TOKEN, left_pad=False)

            processed_data.append((np.array(context), np.array(target)))

    print(f"Total processed data points: {len(processed_data)}")
    return processed_data

# Rest of the code remains the same
training_data_dir = "./bitdata/training"
evaluating_data_dir = "./bitdata/evaluation"

training_file_paths = [
    os.path.join(training_data_dir, f)
    for f in os.listdir(training_data_dir)
    if f.endswith(".json")
]
evaluating_file_paths = [
    os.path.join(evaluating_data_dir, f)
    for f in os.listdir(evaluating_data_dir)
    if f.endswith(".json")
]

# Check if processed data files exist
processed_training_file = "processed_training_data.pkl"
processed_evaluating_file = "processed_evaluating_data.pkl"

if os.path.exists(processed_training_file) and os.path.exists(
    processed_evaluating_file
):
    print("Loading pre-processed data...")
    with open(processed_training_file, 'rb') as f:
        training_data = pickle.load(f)
    with open(processed_evaluating_file, 'rb') as f:
        evaluating_data = pickle.load(f)
    print(f"Loaded {len(training_data)} training data points")
    print(f"Loaded {len(evaluating_data)} evaluation data points")
else:
    print("Processing data...")
    training_data = load_and_process_training_data(
        training_file_paths
    )
    evaluating_data = load_and_process_training_data(
        evaluating_file_paths
    )

    # Save processed data
    with open(processed_training_file, 'wb') as f:
        pickle.dump(training_data, f)
    with open(processed_evaluating_file, 'wb') as f:
        pickle.dump(evaluating_data, f)
    print("Processed data saved.")

print("Data loading completed.")

# Print a few lines to verify the data
print("Training data examples:")
for i in range(min(3, len(training_data))):
    print(f"Input (length {len(training_data[i][0])}): {training_data[i][0]}")
    print(f"Output (length {len(training_data[i][1])}): {training_data[i][1]}")
    print("---")